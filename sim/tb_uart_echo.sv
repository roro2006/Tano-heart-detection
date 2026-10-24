module tb_uart_echo;
    localparam integer CPB = 4;
    logic clk = 0, rst = 1, rx = 1, tx;
    logic [7:0] expected = 8'hA6;
    integer i;
    uart_echo_top #(.CLKS_PER_BIT(CPB)) dut (
        .clk27(clk), .rst(rst), .uart_rx_pin(rx), .uart_tx_pin(tx)
    );
    always #1 clk = ~clk;

    task send_byte(input [7:0] value);
        begin
            rx = 1'b0;
            repeat (CPB) @(negedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                rx = value[i];
                repeat (CPB) @(negedge clk);
            end
            rx = 1'b1;
            repeat (CPB) @(negedge clk);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst = 1'b0;
        send_byte(expected);
    end

    // Echo must begin with a low start bit and contain the original byte;
    // sync on the falling edge and sample each bit in its middle.
    initial begin : check
        integer b;
        fork
            begin
                @(negedge tx);
                repeat (CPB / 2) @(negedge clk);
                if (tx !== 1'b0)
                    $fatal(1, "echo start bit too short");
                for (b = 0; b < 8; b = b + 1) begin
                    repeat (CPB) @(negedge clk);
                    if (tx !== expected[b])
                        $fatal(1, "echo bit %0d mismatch", b);
                end
                repeat (CPB) @(negedge clk);
                if (tx !== 1'b1)
                    $fatal(1, "echo stop bit missing");
            end
            begin
                repeat (CPB * 40) @(negedge clk);
                $fatal(1, "echo did not start");
            end
        join_any
        $display("uart_echo PASS");
        $finish;
    end
endmodule
