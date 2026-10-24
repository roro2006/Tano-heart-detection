module uart_echo_top #(
    parameter integer CLKS_PER_BIT = 234
) (
    input  logic clk27,
    input  logic rst,
    input  logic uart_rx_pin,
    output logic uart_tx_pin
);
    logic [7:0] rx_data;
    logic rx_valid;
    logic tx_start;
    logic tx_busy;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) rx (
        .clk(clk27), .rst(rst), .rx(uart_rx_pin),
        .data(rx_data), .valid(rx_valid)
    );
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) tx (
        .clk(clk27), .rst(rst), .start(tx_start), .data(rx_data),
        .tx(uart_tx_pin), .busy(tx_busy)
    );

    always_ff @(posedge clk27) begin
        if (rst)
            tx_start <= 1'b0;
        else
            tx_start <= rx_valid && !tx_busy;
    end
endmodule
