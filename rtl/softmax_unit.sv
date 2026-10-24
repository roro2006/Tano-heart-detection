// Row softmax over the 16x16 score matrix, bit-exact with softmax_q8_8.
//
// Scores live in the A memory as words of 8 lanes: word (g, j) holds
// S[8g + l][j] in lane l. Each row group g is processed in four passes:
//   MAX  per-lane maximum and first argmax (all 8 lanes in parallel)
//   SUM  per-lane sum of EXP_LUT[min(255, (max - s) >> 4)]
//   DIV  p = (exp << 16) / total, written back as word (g, j) of P
//   FIX  add 65536 - sum(p) to each row's argmax entry
// Lanes are serialised through one ROM and one restoring divider; the whole
// unit takes roughly 9k cycles, which is negligible next to the UART.
module softmax_unit #(
    parameter integer LW = 18,
    parameter logic [8:0] S_BASE = 9'd384,
    parameter logic [8:0] P_BASE = 9'd416
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
        IDLE, MAX_RD, MAX_USE, SUM_RD, SUM_USE, SUM_LANE,
        DIV_RD, DIV_USE, DIV_LANE, DIV_RUN, DIV_WRITE, DIV_DRAIN,
        FIX_RD, FIX_USE, FIX_WAIT
    } state_t;

    state_t state;
    logic g;
    logic [3:0] j;
    logic [2:0] l;
    logic [8*LW-1:0] word;
    logic signed [15:0] row_max [0:7];
    logic [3:0] row_arg [0:7];
    logic [20:0] row_total [0:7];
    logic [20:0] row_psum [0:7];

    logic signed [15:0] lane_score;
    logic [16:0] diff;
    logic [7:0] exp_index;
    logic [16:0] exp_value;

    // Restoring divider: quotient = (exp << 16) / total.
    logic [32:0] div_num;
    logic [32:0] div_quo;
    logic [21:0] div_rem;
    logic [5:0] div_count;
    logic [21:0] div_trial;

    assign lane_score = word[l*LW +: 16];
    assign diff = 17'($signed({row_max[l][15], row_max[l]}) - $signed({lane_score[15], lane_score}));
    assign exp_index = (diff[16:4] > 13'd255) ? 8'd255 : diff[11:4];
    exp_rom rom(.index(exp_index), .value(exp_value));

    assign div_trial = {div_rem[20:0], div_num[32]};

    always_comb begin
        case (state)
            FIX_RD, FIX_USE, FIX_WAIT: raddr = P_BASE + {4'd0, g, row_arg[l]};
            default:                   raddr = S_BASE + {4'd0, g, j};
        endcase
    end

    integer i;
    always_ff @(posedge clk) begin
        done <= 1'b0;
        we <= 1'b0;
        if (rst) begin
            state <= IDLE;
            g <= 1'b0;
            j <= '0;
            l <= '0;
        end else case (state)
            IDLE: if (start) begin
                g <= 1'b0;
                j <= '0;
                state <= MAX_RD;
            end
            MAX_RD: state <= MAX_USE;
            MAX_USE: begin
                for (i = 0; i < 8; i = i + 1)
                    if (j == 0 || $signed(rdata[i*LW +: 16]) > row_max[i]) begin
                        row_max[i] <= rdata[i*LW +: 16];
                        row_arg[i] <= j;
                    end
                j <= j + 1'b1;
                state <= state_t'((j == 4'd15) ? SUM_RD : MAX_RD);
                if (j == 4'd15)
                    for (i = 0; i < 8; i = i + 1) begin
                        row_total[i] <= '0;
                        row_psum[i] <= '0;
                    end
            end
            SUM_RD: state <= SUM_USE;
            SUM_USE: begin
                word <= rdata;
                l <= '0;
                state <= SUM_LANE;
            end
            SUM_LANE: begin
                row_total[l] <= row_total[l] + {4'd0, exp_value};
                l <= l + 1'b1;
                if (l == 3'd7) begin
                    j <= j + 1'b1;
                    state <= state_t'((j == 4'd15) ? DIV_RD : SUM_RD);
                end
            end
            DIV_RD: state <= DIV_USE;
            DIV_USE: begin
                word <= rdata;
                l <= '0;
                state <= DIV_LANE;
            end
            DIV_LANE: begin
                div_num <= {exp_value, 16'd0};
                div_quo <= '0;
                div_rem <= '0;
                div_count <= 6'd33;
                state <= DIV_RUN;
            end
            DIV_RUN: begin
                if (div_count == 0) begin
                    wdata[l*LW +: LW] <= LW'(div_quo);
                    row_psum[l] <= row_psum[l] + div_quo[20:0];
                    l <= l + 1'b1;
                    state <= state_t'((l == 3'd7) ? DIV_WRITE : DIV_LANE);
                end else begin
                    if (div_trial >= {1'b0, row_total[l]}) begin
                        div_rem <= div_trial - {1'b0, row_total[l]};
                        div_quo <= {div_quo[31:0], 1'b1};
                    end else begin
                        div_rem <= div_trial;
                        div_quo <= {div_quo[31:0], 1'b0};
                    end
                    div_num <= {div_num[31:0], 1'b0};
                    div_count <= div_count - 1'b1;
                end
            end
            DIV_WRITE: begin
                we <= 1'b1;
                waddr <= P_BASE + {4'd0, g, j};
                j <= j + 1'b1;
                if (j == 4'd15) begin
                    l <= '0;
                    state <= DIV_DRAIN;
                end else state <= DIV_RD;
            end
            // The registered write of P column 15 lands this cycle; a row whose
            // argmax is 15 must not read that word until it has.
            DIV_DRAIN: state <= FIX_RD;
            FIX_RD: state <= FIX_USE;
            FIX_USE: begin
                // The argmax word was read this cycle; patch only lane l.
                wdata <= rdata;
                wdata[l*LW +: LW] <= rdata[l*LW +: LW] + LW'(21'd65536 - row_psum[l]);
                waddr <= raddr;
                we <= 1'b1;
                state <= FIX_WAIT;
            end
            FIX_WAIT: begin
                // The registered write lands this cycle; two rows may share an
                // argmax word, so the next read must not overlap it.
                l <= l + 1'b1;
                if (l == 3'd7) begin
                    if (g) begin
                        done <= 1'b1;
                        state <= IDLE;
                    end else begin
                        g <= 1'b1;
                        j <= '0;
                        state <= MAX_RD;
                    end
                end else state <= FIX_RD;
            end
            default: state <= IDLE;
        endcase
    end
endmodule
