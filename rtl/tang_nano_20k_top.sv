// Tang Nano 20K (GW2AR-LV18QN88C8/I7) board wrapper, clocked at 27 MHz.
//
// LEDs are active low: 0 heartbeat, 1 busy, 2 computing, 3 UART rx,
// 4 checksum error seen (sticky until reset), 5 lit while held in reset.
// Button S1 resets the design.
module tang_nano_20k_top (
    input  logic clk27,
    input  logic btn_s1,
    input  logic uart_rx,
    output logic uart_tx,
    output logic [5:0] led
);
    localparam integer CLK_HZ = 27_000_000;
    // 27 MHz / 1 Mbaud is exactly 27 clocks per bit.
    localparam integer BAUD = 1_000_000;
    localparam integer STRETCH = 22;

    // Power-on reset held for 2^16 cycles, plus a synchronised button.
    logic [15:0] por_count = '0;
    logic por_done = 1'b0;
    logic [1:0] btn_sync = '0;
    logic rst;
    always_ff @(posedge clk27) begin
        btn_sync <= {btn_sync[0], btn_s1};
        if (!por_done) begin
            por_count <= por_count + 1'b1;
            if (&por_count) por_done <= 1'b1;
        end
    end
    assign rst = !por_done || btn_sync[1];

    logic busy, computing, rx_activity, error_seen;
    tpu_top #(
        .CLKS_PER_BIT((CLK_HZ + BAUD / 2) / BAUD),
        .TIMEOUT_CLKS(CLK_HZ / 10)
    ) accel (
        .clk(clk27), .rst(rst), .uart_rx_pin(uart_rx), .uart_tx_pin(uart_tx),
        .busy(busy), .computing(computing), .rx_activity(rx_activity),
        .error_seen(error_seen));

    // Pulse stretchers so short events are visible.
    logic [24:0] heartbeat = '0;
    logic [STRETCH-1:0] compute_hold = '0;
    logic [STRETCH-1:0] rx_hold = '0;
    always_ff @(posedge clk27) begin
        heartbeat <= heartbeat + 1'b1;
        compute_hold <= computing ? '1 : (compute_hold != 0 ? compute_hold - 1'b1 : '0);
        rx_hold <= rx_activity ? '1 : (rx_hold != 0 ? rx_hold - 1'b1 : '0);
    end

    assign led = ~{rst, error_seen, rx_hold != 0, compute_hold != 0, busy, heartbeat[24]};
endmodule
