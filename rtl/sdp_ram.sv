// Simple dual-port RAM with a registered read port (one BSRAM read cycle).
// Written so yosys infers Gowin SDPB/SDPX9B blocks.
module sdp_ram #(
    parameter integer DEPTH = 512,
    parameter integer WIDTH = 16
) (
    input  logic clk,
    input  logic we,
    input  logic [$clog2(DEPTH)-1:0] waddr,
    input  logic [WIDTH-1:0] wdata,
    input  logic [$clog2(DEPTH)-1:0] raddr,
    output logic [WIDTH-1:0] rdata
);
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end
endmodule
