module uart_rx #(
    parameter integer CLKS_PER_BIT = 234
) (
    input logic clk, input logic rst, input logic rx,
    output logic [7:0] data, output logic valid
);
    integer ticks;
    logic [3:0] bit_index;
    logic [7:0] shift;
    logic active;
    logic rx_meta, rx_sync;
    // Two-flop synchroniser: rx comes straight from the BL616 pin.
    always_ff @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
    end
    always_ff @(posedge clk) begin
        valid <= 1'b0;
        if (rst) begin ticks <= 0; bit_index <= 0; shift <= 0; active <= 1'b0; data <= 0; end
        else if (!active && !rx_sync) begin active <= 1'b1; ticks <= CLKS_PER_BIT + CLKS_PER_BIT/2 - 1; bit_index <= 0; end
        else if (active && ticks == 0) begin
            ticks <= CLKS_PER_BIT-1;
            if (bit_index < 8) begin shift[bit_index] <= rx_sync; bit_index <= bit_index + 1'b1; end
            // Drop bytes whose stop bit is low (framing error).
            else begin data <= shift; valid <= rx_sync; active <= 1'b0; end
        end else if (active) ticks <= ticks - 1;
    end
endmodule
