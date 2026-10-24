`timescale 1ns/1ps
// ln_unit against tpu_model.layer_norm_int (python python/sim_vectors.py --ln).
module tb_ln_unit;
    logic clk = 1'b0, rst = 1'b1, start = 1'b0;
    logic done, we;
    logic [8:0] raddr, waddr;
    logic [143:0] rdata, wdata;
    logic [143:0] mem [0:511];
    logic [143:0] expected [0:127];
    always #5 clk = ~clk;

    ln_unit dut(.clk(clk), .rst(rst), .start(start), .done(done), .raddr(raddr),
                .rdata(rdata), .we(we), .waddr(waddr), .wdata(wdata));
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end

    integer i, l, errors = 0, cycles = 0;
    initial begin
        $readmemh("sim/ln_input.mem", mem);
        $readmemh("sim/ln_expected.mem", expected);
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(negedge clk) start = 1'b1;
        @(negedge clk) start = 1'b0;
        while (!done) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 100_000) $fatal(1, "ln_unit timeout");
        end
        @(posedge clk);
        for (i = 0; i < 128; i = i + 1)
            for (l = 0; l < 8; l = l + 1)
                if (mem[i][l*18 +: 18] !== expected[i][l*18 +: 18]) begin
                    if (errors < 10)
                        $display("row %0d col %0d: got %0d expected %0d", (i / 64) * 8 + l, i % 64,
                                 $signed(mem[i][l*18 +: 18]), $signed(expected[i][l*18 +: 18]));
                    errors = errors + 1;
                end
        if (errors != 0) $fatal(1, "ln_unit FAIL: %0d mismatches", errors);
        $display("ln_unit PASS (%0d cycles for 16 rows)", cycles);
        $finish;
    end
endmodule
