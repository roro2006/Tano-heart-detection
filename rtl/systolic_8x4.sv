// 8x4 output-stationary systolic tile engine.
//
// Row r of A enters on a_edge lane r and column c of B on b_edge lane c. The
// edges must be skewed: A[r][k] is presented at valid cycle k+r and B[k][c]
// at k+c, so PE(r,c) sees the matching pair at cycle k+r+c. A tile with klen
// terms therefore takes klen + 8 + 4 - 2 valid cycles; done pulses on the
// cycle after the last one, when acc holds the final sums. acc stays stable
// until the next clear, so callers read results directly from it.
//
// acc is flattened as acc[(r*4 + c)*ACC_WIDTH +: ACC_WIDTH].
module systolic_8x4 #(
    parameter integer W = 18,
    parameter integer ACC_WIDTH = 40
) (
    input  logic clk,
    input  logic rst,
    input  logic clear,
    input  logic valid_in,
    input  logic shift8,
    input  logic [6:0] klen,
    input  logic [8*W-1:0] a_edge,
    input  logic [4*W-1:0] b_edge,
    output logic done,
    output logic [32*ACC_WIDTH-1:0] acc
);
    localparam integer ROWS = 8;
    localparam integer COLS = 4;

    // a_link[r][c] feeds PE(r,c); column COLS is the unused right-hand exit.
    logic [W-1:0] a_link [0:ROWS-1][0:COLS];
    logic [W-1:0] b_link [0:ROWS][0:COLS-1];
    logic [7:0] count;

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : g_row
            assign a_link[r][0] = a_edge[r*W +: W];
            for (c = 0; c < COLS; c = c + 1) begin : g_col
                if (r == 0) begin : g_top
                    assign b_link[0][c] = b_edge[c*W +: W];
                end
                pe #(.W(W), .ACC_WIDTH(ACC_WIDTH)) u_pe (
                    .clk(clk), .rst(rst), .clear(clear), .valid_in(valid_in),
                    .shift8(shift8),
                    .a_in(a_link[r][c]), .b_in(b_link[r][c]),
                    .a_out(a_link[r][c+1]), .b_out(b_link[r+1][c]),
                    .acc(acc[(r*COLS + c)*ACC_WIDTH +: ACC_WIDTH]));
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rst || clear) begin
            count <= '0;
        end else if (valid_in) begin
            count <= count + 1'b1;
            if (count == {1'b0, klen} + 8'd9)
                done <= 1'b1;
        end
    end
endmodule
