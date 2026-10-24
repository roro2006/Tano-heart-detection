module uart_tx #(
    parameter integer CLKS_PER_BIT = 234
) (
    input logic clk, input logic rst, input logic start,
    input logic [7:0] data, output logic tx, output logic busy
);
    integer ticks;
    logic [3:0] bit_index;
    logic [9:0] shift;
    always_ff @(posedge clk) begin
        if (rst) begin tx <= 1'b1; busy <= 1'b0; ticks <= 0; bit_index <= 0; shift <= '1; end
        else if (start && !busy) begin shift <= {1'b1, data, 1'b0}; busy <= 1'b1; tx <= 1'b0; ticks <= 0; bit_index <= 0; end
        else if (busy) begin
            if (ticks == CLKS_PER_BIT-1) begin
                ticks <= 0; bit_index <= bit_index + 1'b1; shift <= {1'b1, shift[9:1]};
                tx <= shift[1];
                if (bit_index == 4'd9) begin busy <= 1'b0; tx <= 1'b1; end
            end else ticks <= ticks + 1;
        end
    end
endmodule

