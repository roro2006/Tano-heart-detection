// Integer layer norm over 64-wide rows, bit-exact with tpu_model.layer_norm_int.
//
// Rows live in the A memory as row-group words: word (g, k) holds R[8g + l][k]
// in lane l. Each group runs three passes over k = 0..63:
//   SUM  per-lane sum (8 lanes in parallel); mean = sum >>> 6
//   VAR  per-lane sum of c^2, c = r - mean (lanes serialised, one multiplier)
//   OUT  out = sat16((c * inv + 2^(sh-1)) >>> sh), written to word (g, k) of N
// Between VAR and OUT each lane normalises its variance by an even shift into
// [2^16, 2^18) and looks up inv = 1/sqrt in a 768-entry ROM; sh = 13 + k.
module ln_unit #(
    parameter integer LW = 18,
    parameter logic [8:0] R_BASE = 9'd128,
    parameter logic [8:0] N_BASE = 9'd0,
    parameter RSQRT_FILE = "rtl/rsqrt_rom.mem"
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    output logic done,
    output logic [8:0] raddr,
    input  logic [8*LW-1:0] rdata,
    output logic we,
    output logic [8:0] waddr,
    output logic [8*LW-1:0] wdata
);
    typedef enum logic [3:0] {
        IDLE, SUM_RD, SUM_USE, VAR_RD, VAR_USE, VAR_LANE,
        NRM_ADDR, NRM_CAP, OUT_RD, OUT_USE, OUT_LANE, OUT_WRITE, FINISH
    } state_t;

    state_t state;
    logic g;
    logic [5:0] k;
    logic [2:0] l;
    logic [8*LW-1:0] word;
    logic signed [21:0] row_sum [0:7];
    logic signed [15:0] row_mean [0:7];
    logic [40:0] row_var [0:7];
    logic [16:0] row_inv [0:7];
    logic [4:0] row_shift [0:7];

    // 1/sqrt table, synchronous read (one BSRAM).
    logic [16:0] rsqrt [0:1023];
    initial $readmemh(RSQRT_FILE, rsqrt);
    logic [9:0] rom_addr;
    logic [16:0] rom_q;
    always_ff @(posedge clk) rom_q <= rsqrt[rom_addr];

    // Current lane: centred value and the shared multiplier.
    logic signed [15:0] lane_value;
    logic signed [17:0] centred;
    logic signed [17:0] mul_b;
    logic signed [35:0] product;
    assign lane_value = word[l*LW +: 16];
    assign centred = 18'(lane_value) - 18'(row_mean[l]);
    assign mul_b = (state == OUT_LANE) ? $signed({1'b0, row_inv[l]}) : centred;
    assign product = centred * mul_b;

    // Variance normalisation for lane l: k = (bitlen - 17) >>> 1.
    logic [5:0] bitlen;
    logic signed [6:0] even_k;
    logic [40:0] normalised;
    integer b;
    always_comb begin
        bitlen = '0;
        for (b = 0; b < 41; b = b + 1)
            if (row_var[l][b]) bitlen = 6'(b + 1);
        even_k = ($signed({1'b0, bitlen}) - 7'sd17) >>> 1;
        normalised = (even_k >= 0) ? (row_var[l] >> (2 * even_k))
                                   : (row_var[l] << (-2 * even_k));
        // A zero variance lands on the zero padding above entry 767.
        rom_addr = normalised[17:8] - 10'd256;
    end

    // Output rounding for lane l.
    logic signed [36:0] rounded;
    logic signed [36:0] scaled;
    logic signed [15:0] lane_out;
    assign rounded = 37'(product) + (37'sd1 <<< (row_shift[l] - 5'd1));
    assign scaled = rounded >>> row_shift[l];
    assign lane_out = (scaled > 37'sd32767) ? 16'sd32767 :
                      (scaled < -37'sd32768) ? -16'sd32768 : scaled[15:0];

    assign raddr = R_BASE + {2'd0, g, k};

    integer i;
    always_ff @(posedge clk) begin
        done <= 1'b0;
        we <= 1'b0;
        if (rst) begin
            state <= IDLE;
            g <= 1'b0;
            k <= '0;
            l <= '0;
        end else case (state)
            IDLE: if (start) begin
                g <= 1'b0;
                k <= '0;
                for (i = 0; i < 8; i = i + 1) row_sum[i] <= '0;
                state <= SUM_RD;
            end
            SUM_RD: state <= SUM_USE;
            SUM_USE: begin
                for (i = 0; i < 8; i = i + 1)
                    row_sum[i] <= row_sum[i] + 22'($signed(rdata[i*LW +: 16]));
                k <= k + 1'b1;
                if (k == 6'd63) begin
                    state <= VAR_RD;
                    for (i = 0; i < 8; i = i + 1) begin
                        row_mean[i] <= 16'((row_sum[i] + 22'($signed(rdata[i*LW +: 16]))) >>> 6);
                        row_var[i] <= '0;
                    end
                end else state <= SUM_RD;
            end
            VAR_RD: state <= VAR_USE;
            VAR_USE: begin
                word <= rdata;
                l <= '0;
                state <= VAR_LANE;
            end
            VAR_LANE: begin
                row_var[l] <= row_var[l] + 41'(product);
                l <= l + 1'b1;
                if (l == 3'd7) begin
                    k <= k + 1'b1;
                    if (k == 6'd63) state <= NRM_ADDR;
                    else state <= VAR_RD;
                end
            end
            NRM_ADDR: begin
                // rom_addr is presented this cycle; the table answers next cycle.
                row_shift[l] <= 5'(7'sd13 + even_k);
                state <= NRM_CAP;
            end
            NRM_CAP: begin
                row_inv[l] <= rom_q;
                l <= l + 1'b1;
                if (l == 3'd7) state <= OUT_RD;
                else state <= NRM_ADDR;
            end
            OUT_RD: state <= OUT_USE;
            OUT_USE: begin
                word <= rdata;
                l <= '0;
                state <= OUT_LANE;
            end
            OUT_LANE: begin
                wdata[l*LW +: LW] <= LW'(lane_out);
                l <= l + 1'b1;
                if (l == 3'd7) state <= OUT_WRITE;
            end
            OUT_WRITE: begin
                // The registered write lands next cycle, during a read of R.
                we <= 1'b1;
                waddr <= N_BASE + {2'd0, g, k};
                k <= k + 1'b1;
                if (k == 6'd63) begin
                    if (g) state <= FINISH;
                    else begin
                        g <= 1'b1;
                        for (i = 0; i < 8; i = i + 1) row_sum[i] <= '0;
                        state <= SUM_RD;
                    end
                end else state <= OUT_RD;
            end
            FINISH: begin
                // The last word lands this cycle; only now may the core move on.
                done <= 1'b1;
                state <= IDLE;
            end
            default: state <= IDLE;
        endcase
    end
endmodule
