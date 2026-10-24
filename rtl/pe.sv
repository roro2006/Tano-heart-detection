// One registered systolic processing element.
//
// A values flow left to right and B values flow top to bottom. The element
// multiplies the operands it receives this cycle, so a_out/b_out are the
// one-cycle-delayed copies seen by its right and lower neighbours.
// shift8 selects the Q8.8 product rounding used by dot_q8_8 in the Python
// oracle: every product is floor-shifted by 8 before it is accumulated.
module pe #(
    parameter integer W = 18,
    parameter integer ACC_WIDTH = 40
) (
    input  logic clk,
    input  logic rst,
    input  logic clear,
    input  logic valid_in,
    input  logic shift8,
    input  logic signed [W-1:0] a_in,
    input  logic signed [W-1:0] b_in,
    output logic signed [W-1:0] a_out,
    output logic signed [W-1:0] b_out,
    output logic signed [ACC_WIDTH-1:0] acc
);
    logic signed [2*W-1:0] product;
    logic signed [2*W-1:0] term;

    assign product = a_in * b_in;
    assign term = shift8 ? (product >>> 8) : product;

    always_ff @(posedge clk) begin
        if (rst || clear) begin
            a_out <= '0;
            b_out <= '0;
            acc <= '0;
        end else if (valid_in) begin
            a_out <= a_in;
            b_out <= b_in;
            acc <= acc + ACC_WIDTH'(term);
        end
    end
endmodule
