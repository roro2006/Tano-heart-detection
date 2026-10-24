// The on-chip model sequencer, bit-exact with python/tpu_model.py.
//
// Programs:
//   PROG_ATTN   one attention layer, O = ATT Wo written to X (attention_q8_8)
//   PROG_INFER  the full model: embed, 2 transformer layers, output head,
//               argmax (tpu_model.forward / predicted_class)
//
// Memories (lane 0 in the low bits of every word):
//   A  512 x (8 x 18)  row-group words: word (g, k) holds M[8g + l][k] in lane l
//      X/N 0..127   Q/R 128..255   ATT/H 256..383   S/logits 384..415
//      P 416..447   F (features) 448..479
//   W  12800 x (4 x int8) weights, owned by tpu_top (see tpu_model.weight_words);
//      lane value is int8 << lsh, lsh per slot from w_lsh
//   C  512 x (4 x 16) column-group words, owned here:
//      K^T 0..255 (cg*64 + feature, lanes = tokens)
//      V   256..511 (256 + cg*16 + token, lanes = features)
//   POS 128 x (8 x 16) position table in A layout, owned by tpu_top
//
// Each matmul runs 2 row groups x (cg_last + 1) column groups of 8x4 tiles.
// Writeback saturates to 16 bits, then optionally applies ReLU and adds a
// residual word (from A or POS) with a second saturation.
module tpu_core #(
    parameter integer ACC_WIDTH = 40
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic prog,
    input  logic layer,
    output logic busy,
    output logic done,
    output logic [3:0] class_out,
    output logic [31:0] cycles,
    output logic [8:0] a_raddr,
    input  logic [143:0] a_rdata,
    output logic a_we,
    output logic [8:0] a_waddr,
    output logic [143:0] a_wdata,
    output logic [13:0] w_raddr,
    input  logic [31:0] w_rdata,
    output logic [3:0] w_slot,
    input  logic [2:0] w_lsh,
    output logic [6:0] pos_raddr,
    input  logic [127:0] pos_rdata
);
    localparam integer LW = 18;
    localparam logic PROG_ATTN = 1'b0, PROG_INFER = 1'b1;
    localparam logic [8:0] A_X = 9'd0, A_Q = 9'd128, A_ATT = 9'd256,
                           A_S = 9'd384, A_P = 9'd416, A_F = 9'd448;
    localparam logic [8:0] C_KT = 9'd0, C_V = 9'd256;
    localparam logic [1:0] PH_EMBED = 2'd0, PH_LAYER = 2'd1, PH_OUTPUT = 2'd2, PH_ARGMAX = 2'd3;
    localparam logic [3:0] S_Q = 4'd0, S_K = 4'd1, S_V = 4'd2, S_S = 4'd3, S_SM = 4'd4,
                           S_ATT = 4'd5, S_O = 4'd6, S_LN = 4'd7, S_W1 = 4'd8, S_W2 = 4'd9;
    localparam logic [1:0] K_MATMUL = 2'd0, K_SOFTMAX = 2'd1, K_LNORM = 2'd2, K_ARGMAX = 2'd3;
    localparam logic [1:0] OUT_A = 2'd0, OUT_KT = 2'd1, OUT_V = 2'd2;
    localparam logic [1:0] WB_READ = 2'd0, WB_LANES = 2'd1, WB_WRITE = 2'd2;

    typedef enum logic [3:0] {
        IDLE, OP_SETUP, TILE, WRITEBACK, SOFTMAX_RUN, LN_RUN,
        AM_RD0, AM_RD1, AM_USE, NEXT_OP, FINISH
    } state_t;
    state_t state;

    // Program position.
    logic prog_r, cur_layer;
    logic [1:0] phase;
    logic [3:0] step;

    // Op decode.
    logic [1:0] kind;
    logic [8:0] a_base, out_base, res_base;
    logic k16, out_k16, b_from_w, b_stride16, relu, res_en, res_pos, shift8;
    logic [3:0] slot, cg_last;
    logic [8:0] c_base;
    logic [1:0] out_mode;
    logic [4:0] post_shift;
    logic [3:0] layer_slot;

    assign layer_slot = cur_layer ? 4'd6 : 4'd0;

    always_comb begin
        kind = K_MATMUL; a_base = A_X; k16 = 1'b0; b_from_w = 1'b1; slot = layer_slot;
        c_base = C_KT; b_stride16 = 1'b0; cg_last = 4'd15; out_mode = OUT_A; out_base = A_Q;
        out_k16 = 1'b0; post_shift = 5'd0; shift8 = 1'b1; relu = 1'b0; res_en = 1'b0;
        res_pos = 1'b0; res_base = A_X;
        case (phase)
            PH_EMBED: begin
                a_base = A_F; k16 = 1'b1; slot = 4'd12; b_stride16 = 1'b1; out_base = A_X;
                res_en = 1'b1; res_pos = 1'b1;
            end
            PH_OUTPUT: begin slot = 4'd13; cg_last = 4'd3; out_base = A_S; out_k16 = 1'b1; end
            PH_ARGMAX: kind = K_ARGMAX;
            default: case (step)
                S_K: begin slot = layer_slot + 4'd1; out_mode = OUT_KT; end
                S_V: begin slot = layer_slot + 4'd2; out_mode = OUT_V; end
                S_S: begin
                    a_base = A_Q; b_from_w = 1'b0; c_base = C_KT; cg_last = 4'd3;
                    out_base = A_S; out_k16 = 1'b1; post_shift = 5'd3;
                end
                S_SM: kind = K_SOFTMAX;
                S_ATT: begin
                    a_base = A_P; k16 = 1'b1; b_from_w = 1'b0; c_base = C_V; b_stride16 = 1'b1;
                    out_base = A_ATT; post_shift = 5'd16; shift8 = 1'b0;
                end
                S_O: begin
                    a_base = A_ATT; slot = layer_slot + 4'd3;
                    // Attention alone writes O over X; the full model writes R = X + O.
                    out_base = (prog_r == PROG_INFER) ? A_Q : A_X;
                    res_en = (prog_r == PROG_INFER);
                end
                S_LN: kind = K_LNORM;
                S_W1: begin slot = layer_slot + 4'd4; out_base = A_ATT; relu = 1'b1; end
                S_W2: begin a_base = A_ATT; slot = layer_slot + 4'd5; out_base = A_X; res_en = 1'b1; end
                default: ;   // S_Q
            endcase
        endcase
    end

    // Tile iteration state.
    logic g;
    logic [3:0] cg;
    logic [6:0] n;
    logic [2:0] wb;
    logic [1:0] wb_step;
    logic rd_valid;
    logic [6:0] kdim;
    assign kdim = k16 ? 7'd16 : 7'd64;

    // B operand addresses: column group cg, term n.
    logic [9:0] cg_offset;
    logic [13:0] slot_base;
    assign cg_offset = b_stride16 ? {2'd0, cg, 4'd0} : {cg, 6'd0};
    assign slot_base = (slot == 4'd12) ? 14'd12288 : (slot == 4'd13) ? 14'd12544 : {slot, 10'd0};
    assign w_raddr = slot_base + {4'd0, cg_offset} + {7'd0, n};
    assign w_slot = slot;

    logic c_we;
    logic [8:0] c_waddr, c_raddr;
    logic [63:0] c_wdata, c_rdata;
    assign c_raddr = c_base + cg_offset[8:0] + {2'd0, n};
    sdp_ram #(.DEPTH(512), .WIDTH(64)) cmem(
        .clk(clk), .we(c_we), .waddr(c_waddr), .wdata(c_wdata),
        .raddr(c_raddr), .rdata(c_rdata));

    // Operand skew: lane l of A is delayed l cycles, lane c of B c cycles.
    logic b_sel_w;
    logic [143:0] a_gated;
    logic [71:0] b_gated;
    logic [143:0] a_skew [0:6];
    logic [71:0] b_skew [0:2];
    logic [143:0] a_edge;
    logic [71:0] b_edge;

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : g_a
            assign a_gated[lane*LW +: LW] = rd_valid ? a_rdata[lane*LW +: LW] : '0;
            if (lane == 0) begin : g_a0
                assign a_edge[0 +: LW] = a_gated[0 +: LW];
            end else begin : g_an
                assign a_edge[lane*LW +: LW] = a_skew[lane-1][lane*LW +: LW];
            end
        end
        for (lane = 0; lane < 4; lane = lane + 1) begin : g_b
            logic signed [LW-1:0] w_value;
            logic signed [15:0] c_value;
            assign w_value = LW'($signed(w_rdata[lane*8 +: 8])) <<< w_lsh;
            assign c_value = c_rdata[lane*16 +: 16];
            assign b_gated[lane*LW +: LW] = !rd_valid ? '0 : b_sel_w ? w_value : LW'(c_value);
            if (lane == 0) begin : g_b0
                assign b_edge[0 +: LW] = b_gated[0 +: LW];
            end else begin : g_bn
                assign b_edge[lane*LW +: LW] = b_skew[lane-1][lane*LW +: LW];
            end
        end
    endgenerate

    logic array_clear, array_valid, array_done;
    logic [32*ACC_WIDTH-1:0] acc;
    systolic_8x4 #(.W(LW), .ACC_WIDTH(ACC_WIDTH)) array(
        .clk(clk), .rst(rst), .clear(array_clear), .valid_in(array_valid),
        .shift8(shift8), .klen(kdim), .a_edge(a_edge), .b_edge(b_edge),
        .done(array_done), .acc(acc));

    // Tile writeback, one result per cycle through a single post-processing
    // path: sat16(acc >>> post_shift), ReLU, then the residual lane read at the
    // start of the word (A memory or POS) with a second saturation.
    // Word wb gathers lanes wb_lane = 0 .. wb_lane_last, then is written.
    logic [2:0] wb_last, wb_lane, wb_lane_last;
    logic [8:0] wb_offset, wb_addr;
    logic [4:0] wb_pick;
    always_comb begin
        wb_last = (out_mode == OUT_A) ? 3'd3 : 3'd7;
        wb_lane_last = (out_mode == OUT_A) ? 3'd7 : 3'd3;
        wb_offset = (g ? (out_k16 ? 9'd16 : 9'd64) : 9'd0) + {3'd0, cg, wb[1:0]};
        case (out_mode)
            OUT_KT: begin
                // Rows 4h..4h+3 of the tile are tokens 4(2g+h)..; column i is a feature.
                wb_addr = C_KT + {1'b0, g, wb[2], 6'd0} + {3'd0, cg, wb[1:0]};
                wb_pick = {wb[2], wb_lane[1:0], wb[1:0]};
            end
            OUT_V: begin
                wb_addr = C_V + {1'b0, cg, 4'd0} + {5'd0, g, wb};
                wb_pick = {wb, wb_lane[1:0]};
            end
            default: begin
                wb_addr = out_base + wb_offset;
                wb_pick = {wb_lane, wb[1:0]};
            end
        endcase
    end

    logic signed [ACC_WIDTH-1:0] pick_raw, pick_shifted;
    logic signed [15:0] pick_sat, pick_rect, pick_residual, pick_summed, pick_lane;
    logic signed [16:0] pick_total;
    assign pick_raw = acc[wb_pick*ACC_WIDTH +: ACC_WIDTH];
    assign pick_shifted = pick_raw >>> post_shift;
    assign pick_sat = (pick_shifted > 32767) ? 16'sd32767 :
                      (pick_shifted < -32768) ? -16'sd32768 : pick_shifted[15:0];
    assign pick_rect = (relu && pick_sat < 0) ? 16'sd0 : pick_sat;
    assign pick_residual = res_pos ? pos_rdata[wb_lane*16 +: 16] : a_rdata[wb_lane*LW +: 16];
    assign pick_total = 17'(pick_rect) + 17'(pick_residual);
    // Saturate into a signed wire: a bare pick_total[15:0] is unsigned and
    // would zero-extend negative lanes to 18 bits.
    assign pick_summed = (pick_total > 17'sd32767) ? 16'sd32767 :
                         (pick_total < -17'sd32768) ? -16'sd32768 : pick_total[15:0];
    assign pick_lane = res_en ? pick_summed : pick_rect;

    logic [143:0] wb_a_data;
    logic [63:0] wb_c_data;

    // Softmax and layer norm share the A memory ports while they run.
    logic sm_start, sm_done, sm_we, ln_start, ln_done, ln_we;
    logic [8:0] sm_raddr, sm_waddr, ln_raddr, ln_waddr;
    logic [143:0] sm_wdata, ln_wdata;
    softmax_unit #(.LW(LW), .S_BASE(A_S), .P_BASE(A_P)) softmax(
        .clk(clk), .rst(rst), .start(sm_start), .done(sm_done),
        .raddr(sm_raddr), .rdata(a_rdata), .we(sm_we), .waddr(sm_waddr),
        .wdata(sm_wdata));
    ln_unit #(.LW(LW), .R_BASE(A_Q), .N_BASE(A_X)) lnorm(
        .clk(clk), .rst(rst), .start(ln_start), .done(ln_done),
        .raddr(ln_raddr), .rdata(a_rdata), .we(ln_we), .waddr(ln_waddr),
        .wdata(ln_wdata));

    // Argmax over the summed logits: column j's total is the sum of all
    // 16 lanes of words (0, j) and (1, j) in the logits region.
    logic [3:0] am_j;
    logic signed [19:0] am_half, am_best;
    logic signed [18:0] lane_sum;
    integer s;
    always_comb begin
        lane_sum = '0;
        for (s = 0; s < 8; s = s + 1)
            lane_sum = lane_sum + 19'($signed(a_rdata[s*LW +: 16]));
    end

    logic wb_write;
    assign wb_write = (state == WRITEBACK) && wb_step == WB_WRITE;
    always_comb begin
        case (state)
            SOFTMAX_RUN: a_raddr = sm_raddr;
            LN_RUN:      a_raddr = ln_raddr;
            AM_RD0:      a_raddr = A_S + {5'd0, am_j};
            AM_RD1:      a_raddr = A_S + 9'd16 + {5'd0, am_j};
            WRITEBACK:   a_raddr = res_base + wb_offset;
            default:     a_raddr = a_base + (g ? 9'(kdim) : 9'd0) + {2'd0, n};
        endcase
        case (state)
            SOFTMAX_RUN: begin a_we = sm_we; a_waddr = sm_waddr; a_wdata = sm_wdata; end
            LN_RUN:      begin a_we = ln_we; a_waddr = ln_waddr; a_wdata = ln_wdata; end
            default: begin
                a_we = wb_write && out_mode == OUT_A;
                a_waddr = wb_addr;
                a_wdata = wb_a_data;
            end
        endcase
    end
    assign pos_raddr = wb_offset[6:0];
    assign c_we = wb_write && out_mode != OUT_A;
    assign c_waddr = wb_addr;
    assign c_wdata = wb_c_data;

    integer t;
    always_ff @(posedge clk) begin
        done <= 1'b0;
        sm_start <= 1'b0;
        ln_start <= 1'b0;
        array_clear <= 1'b0;
        array_valid <= 1'b0;
        // Read data for address n appears one cycle later.
        rd_valid <= (state == TILE) && (n < kdim);
        b_sel_w <= b_from_w;
        a_skew[0] <= a_gated;
        b_skew[0] <= b_gated;
        for (t = 1; t < 7; t = t + 1) a_skew[t] <= a_skew[t-1];
        for (t = 1; t < 3; t = t + 1) b_skew[t] <= b_skew[t-1];
        if (busy) cycles <= cycles + 1'b1;

        if (rst) begin
            state <= IDLE;
            busy <= 1'b0;
            prog_r <= PROG_ATTN;
            cur_layer <= 1'b0;
            phase <= PH_LAYER;
            step <= S_Q;
            g <= 1'b0;
            cg <= '0;
            n <= '0;
            wb <= '0;
            wb_step <= WB_READ;
            rd_valid <= 1'b0;
            class_out <= '0;
            cycles <= '0;
        end else case (state)
            IDLE: if (start) begin
                busy <= 1'b1;
                cycles <= '0;
                prog_r <= prog;
                cur_layer <= (prog == PROG_INFER) ? 1'b0 : layer;
                phase <= (prog == PROG_INFER) ? PH_EMBED : PH_LAYER;
                step <= S_Q;
                state <= OP_SETUP;
            end
            OP_SETUP: begin
                g <= 1'b0;
                cg <= '0;
                n <= '0;
                case (kind)
                    K_SOFTMAX: begin sm_start <= 1'b1; state <= SOFTMAX_RUN; end
                    K_LNORM:   begin ln_start <= 1'b1; state <= LN_RUN; end
                    K_ARGMAX:  begin am_j <= '0; state <= AM_RD0; end
                    default:   begin array_clear <= 1'b1; state <= TILE; end
                endcase
            end
            TILE: begin
                // Cycle n = 0 issues the first read while the array clears;
                // cycles n = 1 .. kdim + 10 carry the skewed edges.
                array_valid <= (n <= kdim + 7'd9);
                n <= n + 1'b1;
                if (array_done) begin
                    wb <= '0;
                    wb_step <= WB_READ;
                    state <= WRITEBACK;
                end
            end
            WRITEBACK: case (wb_step)
                // The residual word's address is up; its data arrives next cycle.
                WB_READ: begin
                    wb_lane <= '0;
                    wb_step <= WB_LANES;
                end
                WB_LANES: begin
                    wb_a_data[wb_lane*LW +: LW] <= LW'(pick_lane);
                    wb_c_data[wb_lane[1:0]*16 +: 16] <= pick_lane;
                    wb_lane <= wb_lane + 1'b1;
                    if (wb_lane == wb_lane_last) wb_step <= WB_WRITE;
                end
                default: begin
                    // WB_WRITE: word wb is written on this edge.
                    wb_step <= WB_READ;
                    if (wb != wb_last) begin
                        wb <= wb + 1'b1;
                    end else begin
                        n <= '0;
                        if (cg != cg_last) begin
                            cg <= cg + 1'b1;
                            array_clear <= 1'b1;
                            state <= TILE;
                        end else if (!g) begin
                            cg <= '0;
                            g <= 1'b1;
                            array_clear <= 1'b1;
                            state <= TILE;
                        end else state <= NEXT_OP;
                    end
                end
            endcase
            SOFTMAX_RUN: if (sm_done) state <= NEXT_OP;
            LN_RUN: if (ln_done) state <= NEXT_OP;
            AM_RD0: state <= AM_RD1;
            AM_RD1: begin
                am_half <= 20'(lane_sum);
                state <= AM_USE;
            end
            AM_USE: begin
                if (am_j == 0 || am_half + 20'(lane_sum) > am_best) begin
                    am_best <= am_half + 20'(lane_sum);
                    class_out <= am_j;
                end
                am_j <= am_j + 1'b1;
                if (am_j == 4'd15) state <= NEXT_OP;
                else state <= AM_RD0;
            end
            NEXT_OP: begin
                state <= OP_SETUP;
                case (phase)
                    PH_EMBED: begin phase <= PH_LAYER; step <= S_Q; end
                    PH_LAYER: begin
                        if (prog_r == PROG_ATTN && step == S_O) state <= FINISH;
                        else if (step != S_W2) step <= step + 1'b1;
                        else if (cur_layer) phase <= PH_OUTPUT;
                        else begin cur_layer <= 1'b1; step <= S_Q; end
                    end
                    PH_OUTPUT: phase <= PH_ARGMAX;
                    default: state <= FINISH;
                endcase
            end
            FINISH: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= IDLE;
            end
            default: state <= IDLE;
        endcase
    end
endmodule
