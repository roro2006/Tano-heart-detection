`timescale 1ns/1ps
// End-to-end UART test: replays sim/tpu_top_vectors.mem (python/sim_vectors.py)
// through tpu_top with a fast bit time and checks every response byte.
module tb_tpu_top;
    localparam integer CPB = 8;
    logic clk = 1'b0, rst = 1'b1, rx = 1'b1;
    wire tx;
    tpu_top #(.CLKS_PER_BIT(CPB), .TIMEOUT_CLKS(100_000)) dut(
        .clk(clk), .rst(rst), .uart_rx_pin(rx), .uart_tx_pin(tx),
        .busy(), .computing(), .rx_activity(), .error_seen());
    always #5 clk = ~clk;

    logic [11:0] script [0:131071];
    logic [7:0] rx_queue [0:65535];
    integer rx_wr = 0, rx_rd = 0, errors = 0, checked = 0;

    // UART receiver for the DUT's tx line.
    initial begin : monitor
        integer b;
        logic [7:0] value;
        forever begin
            @(negedge tx);
            repeat (CPB / 2) @(posedge clk);
            if (tx !== 1'b0) $fatal(1, "tx start bit glitch");
            for (b = 0; b < 8; b = b + 1) begin
                repeat (CPB) @(posedge clk);
                value[b] = tx;
            end
            repeat (CPB) @(posedge clk);
            if (tx !== 1'b1) $fatal(1, "tx stop bit missing");
            rx_queue[rx_wr] = value;
            rx_wr = rx_wr + 1;
        end
    end

    task automatic send_byte(input logic [7:0] value);
        integer b;
        rx = 1'b0;
        repeat (CPB) @(posedge clk);
        for (b = 0; b < 8; b = b + 1) begin
            rx = value[b];
            repeat (CPB) @(posedge clk);
        end
        rx = 1'b1;
        repeat (CPB) @(posedge clk);
    endtask

    initial begin : watchdog
        #400_000_000;
        $fatal(1, "timeout: %0d bytes checked", checked);
    end

    initial begin
        integer i, wait_cycles;
        $readmemh("sim/tpu_top_vectors.mem", script);
        repeat (20) @(posedge clk);
        rst = 1'b0;
        repeat (20) @(posedge clk);
        for (i = 0; script[i] !== 12'hFFF; i = i + 1) begin
            // 0xx: send, 1xx: expect xx, 2xx: expect any byte (cycle counts).
            if (script[i][9:8] == 2'b00) begin
                send_byte(script[i][7:0]);
            end else begin
                wait_cycles = 0;
                while (rx_rd >= rx_wr) begin
                    @(posedge clk);
                    wait_cycles = wait_cycles + 1;
                    if (wait_cycles > 200_000)
                        $fatal(1, "no response byte at script entry %0d", i);
                end
                if (script[i][9:8] == 2'b01 && rx_queue[rx_rd] !== script[i][7:0]) begin
                    if (errors < 20)
                        $display("mismatch at script %0d (response byte %0d): got %02x expected %02x",
                                 i, rx_rd, rx_queue[rx_rd], script[i][7:0]);
                    errors = errors + 1;
                end
                rx_rd = rx_rd + 1;
                checked = checked + 1;
            end
        end
        repeat (CPB * 40) @(posedge clk);
        if (rx_wr != rx_rd) begin
            $display("unexpected extra response bytes: %0d", rx_wr - rx_rd);
            errors = errors + 1;
        end
        if (errors != 0) $fatal(1, "tpu_top FAIL: %0d errors in %0d bytes", errors, checked);
        $display("tpu_top PASS (%0d response bytes checked)", checked);
        $finish;
    end
endmodule
