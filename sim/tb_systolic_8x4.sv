`timescale 1ns/1ps
// Random 8xK by Kx4 tiles through the skewed edges, in both product modes,
// checked against a behavioural sum (every PE, including the far corner).
module tb_systolic_8x4;
    localparam integer W = 18, ACC = 40;
    logic clk = 1'b0, rst = 1'b1, clear = 1'b0, valid_in = 1'b0, shift8 = 1'b0, done;
    logic [6:0] klen;
    logic [8*W-1:0] a_edge;
    logic [4*W-1:0] b_edge;
    logic [32*ACC-1:0] acc;
    systolic_8x4 #(.W(W), .ACC_WIDTH(ACC)) dut(
        .clk(clk), .rst(rst), .clear(clear), .valid_in(valid_in), .shift8(shift8),
        .klen(klen), .a_edge(a_edge), .b_edge(b_edge), .done(done), .acc(acc));
    always #5 clk = ~clk;

    integer a [0:7][0:63];
    integer b [0:63][0:3];
    integer r, c, k, t, trial, errors = 0;
    longint expected;

    initial begin
        repeat (3) @(negedge clk);
        rst = 1'b0;
        for (trial = 0; trial < 6; trial = trial + 1) begin
            klen = (trial % 2) ? 7'd16 : 7'd64;
            shift8 = trial < 4;
            for (r = 0; r < 8; r = r + 1)
                for (k = 0; k < klen; k = k + 1)
                    a[r][k] = (trial == 5) ? 65536 : $signed($random) % 4096;
            for (k = 0; k < klen; k = k + 1)
                for (c = 0; c < 4; c = c + 1)
                    b[k][c] = $signed($random) % 32768;
            @(negedge clk); clear = 1'b1;
            @(negedge clk); clear = 1'b0;
            for (t = 0; t < klen + 10; t = t + 1) begin
                for (r = 0; r < 8; r = r + 1)
                    a_edge[r*W +: W] = (t >= r && t - r < klen) ? W'(a[r][t-r]) : '0;
                for (c = 0; c < 4; c = c + 1)
                    b_edge[c*W +: W] = (t >= c && t - c < klen) ? W'(b[t-c][c]) : '0;
                valid_in = 1'b1;
                @(negedge clk);
            end
            valid_in = 1'b0;
            if (!done) $fatal(1, "done did not pulse after klen + 10 cycles");
            @(negedge clk);
            if (done) $fatal(1, "done stayed high");
            for (r = 0; r < 8; r = r + 1)
                for (c = 0; c < 4; c = c + 1) begin
                    expected = 0;
                    for (k = 0; k < klen; k = k + 1)
                        expected = expected + (shift8 ? ((longint'(a[r][k]) * b[k][c]) >>> 8)
                                                      : longint'(a[r][k]) * b[k][c]);
                    if ($signed(acc[(r*4 + c)*ACC +: ACC]) !== expected) begin
                        $display("trial %0d PE(%0d,%0d): got %0d expected %0d", trial, r, c,
                                 $signed(acc[(r*4 + c)*ACC +: ACC]), expected);
                        errors = errors + 1;
                    end
                end
        end
        if (errors != 0) $fatal(1, "systolic_8x4 FAIL (%0d)", errors);
        $display("systolic_8x4 PASS");
        $finish;
    end
endmodule
