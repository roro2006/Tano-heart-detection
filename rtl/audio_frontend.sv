// Streaming heart-sound front end, bit-exact with python/heart_features.py.
//
// Per sample (2 kHz PCM), 16 bandpass biquads in direct form I with 2
// fractional bits of state, three cycles per band on three multipliers:
//     y = sat18((4 g (x - x2) - a1 y1 - a2 y2 + 2^15) >>> 16),  A[b] += |y|
// Every 256 samples each band's A becomes L = log2_q8_8(A) in a 16-frame
// ring. When frame e (0-based) completes with e = 17, 25, 33, ... (a 2-frame
// warm-up, then a hop of 8), the window of frames e-15 .. e is written to the
// core's feature region as F[t][b] = L - (sum of the 256 L >> 8): word
// (g, band) holds token t = 8g + l in lane l. window_ready then stays high
// until resume, and no samples are accepted meanwhile.
//
// Per-band state {A, y2, y1} lives in a 16-entry RAM rather than registers:
// it is only ever touched one band at a time, and a RAM avoids 16-way muxes.
module audio_frontend #(
    parameter logic [8:0] F_BASE = 9'd448
) (
    input  logic clk,
    input  logic rst,
    input  logic stream_reset,
    input  logic signed [15:0] sample,
    input  logic sample_valid,
    output logic sample_ready,
    output logic window_ready,
    input  logic resume,
    output logic a_we,
    output logic [8:0] a_waddr,
    output logic [143:0] a_wdata
);
    localparam integer LW = 18;
    typedef enum logic [3:0] {
        CLEAR, IDLE, BAND_RD, BAND_MUL, BAND_ACC, FRAME_RD, FRAME_LUT, FRAME_LOG,
        SUM, WIN_RD, WIN_CAP, WIN_WRITE, WIN_FLUSH, WAIT
    } state_t;
    state_t state;

    logic signed [15:0] x1, x2;
    logic signed [16:0] diff;
    logic [3:0] b;
    logic [7:0] pos;
    logic [15:0] frames;
    logic warm;

    // Band state RAM: {energy[25:0], y2[17:0], y1[17:0]}.
    logic st_we;
    logic [61:0] st_wdata, st_rdata;
    sdp_ram #(.DEPTH(16), .WIDTH(62)) band_state(
        .clk(clk), .we(st_we), .waddr(b), .wdata(st_wdata), .raddr(b), .rdata(st_rdata));
    logic [25:0] st_energy;
    logic signed [17:0] st_y1, st_y2;
    assign {st_energy, st_y2, st_y1} = st_rdata;

    logic signed [17:0] coef_g, coef_a1, coef_a2;
    logic [7:0] frac_index, frac;
    heart_roms roms(.band(b), .g(coef_g), .a1(coef_a1), .a2(coef_a2),
                    .frac_index(frac_index), .frac(frac));
    // ROM outputs are registered so the tables map to block RAM: coefficients
    // for band b are ready one cycle after b is set (BAND_RD), the log2
    // fraction one cycle after the energy arrives (FRAME_LUT).
    logic signed [17:0] g_q, a1_q, a2_q;
    logic [7:0] frac_q;
    always_ff @(posedge clk) begin
        g_q <= coef_g;
        a1_q <= coef_a1;
        a2_q <= coef_a2;
        frac_q <= frac;
    end

    // Products are registered between BAND_MUL and BAND_ACC.
    logic signed [34:0] p_g;
    logic signed [35:0] p_a1, p_a2;
    logic signed [39:0] acc_sum, acc_shifted;
    logic signed [17:0] y_new;
    logic [18:0] y_abs;
    assign acc_sum = (40'(p_g) <<< 2) - 40'(p_a1) - 40'(p_a2) + 40'sd32768;
    assign acc_shifted = acc_sum >>> 16;
    assign y_new = (acc_shifted > 40'sd131071) ? 18'sd131071 :
                   (acc_shifted < -40'sd131072) ? -18'sd131072 : acc_shifted[17:0];
    // 19 bits: |-131072| does not fit the 18-bit state.
    assign y_abs = y_new[17] ? 19'(-$signed({y_new[17], y_new})) : {1'b0, y_new};

    // log2_q8_8 of the band's energy: MSB position and the next 8 bits.
    logic [4:0] msb;
    logic [12:0] log_value;
    integer i;
    always_comb begin
        msb = '0;
        for (i = 0; i < 26; i = i + 1)
            if (st_energy[i]) msb = 5'(i);
        frac_index = (msb >= 5'd8) ? 8'(st_energy >> (msb - 5'd8)) : 8'(st_energy << (5'd8 - msb));
        log_value = (st_energy == 0) ? 13'd0 : {msb, frac_q};
    end

    always_comb begin
        case (state)
            CLEAR:     begin st_we = 1'b1; st_wdata = '0; end
            BAND_ACC:  begin st_we = 1'b1; st_wdata = {st_energy + 26'(y_abs), st_y1, y_new}; end
            FRAME_LOG: begin st_we = 1'b1; st_wdata = {26'd0, st_y2, st_y1}; end
            default:   begin st_we = 1'b0; st_wdata = st_rdata; end
        endcase
    end

    // Ring of the last 16 frames: address {frame slot, band}.
    logic ring_we;
    logic [7:0] ring_waddr, ring_raddr;
    logic [15:0] ring_wdata, ring_rdata;
    sdp_ram #(.DEPTH(256), .WIDTH(16)) ring(
        .clk(clk), .we(ring_we), .waddr(ring_waddr), .wdata(ring_wdata),
        .raddr(ring_raddr), .rdata(ring_rdata));
    assign ring_we = (state == FRAME_LOG);
    assign ring_waddr = {frames[3:0], b};
    assign ring_wdata = {3'd0, log_value};

    // Window assembly.
    logic [8:0] sum_count;
    logic [23:0] log_sum;
    logic [15:0] mean;
    logic [3:0] first_slot;
    logic win_g;
    logic [3:0] win_k;
    logic [2:0] win_l;
    logic [143:0] word;
    logic signed [16:0] centred;
    assign centred = $signed({1'b0, ring_rdata}) - $signed({1'b0, mean});
    always_comb begin
        case (state)
            SUM: ring_raddr = sum_count[7:0];
            default: ring_raddr = {first_slot + {win_g, win_l}, win_k};
        endcase
    end

    assign sample_ready = (state == IDLE);
    assign window_ready = (state == WAIT);

    always_ff @(posedge clk) begin
        a_we <= 1'b0;
        if (rst || stream_reset) begin
            // CLEAR zeroes the 16 band states before any sample is taken.
            state <= CLEAR;
            x1 <= '0;
            x2 <= '0;
            b <= '0;
            pos <= '0;
            frames <= '0;
            warm <= 1'b0;
        end else case (state)
            CLEAR: begin
                b <= b + 1'b1;
                if (b == 4'd15) state <= IDLE;
            end
            IDLE: if (sample_valid) begin
                diff <= 17'(sample) - 17'(x2);
                x2 <= x1;
                x1 <= sample;
                b <= '0;
                state <= BAND_RD;
            end
            BAND_RD: state <= BAND_MUL;          // band b's state arrives next cycle
            BAND_MUL: begin
                p_g <= g_q * diff;
                p_a1 <= a1_q * st_y1;
                p_a2 <= a2_q * st_y2;
                state <= BAND_ACC;
            end
            BAND_ACC: begin                      // st_we writes the updated state
                b <= b + 1'b1;
                if (b != 4'd15) state <= BAND_RD;
                else begin
                    pos <= pos + 1'b1;
                    state <= state_t'((pos == 8'd255) ? FRAME_RD : IDLE);
                end
            end
            FRAME_RD: state <= FRAME_LUT;          // band b's energy arrives next cycle
            FRAME_LUT: state <= FRAME_LOG;         // its log2 fraction one cycle later
            FRAME_LOG: begin
                // ring_we stores log_value; st_we clears the band's energy.
                b <= b + 1'b1;
                if (b != 4'd15) state <= FRAME_RD;
                else begin
                    frames <= frames + 1'b1;
                    if (frames + 16'd1 == 16'd18) warm <= 1'b1;
                    if ((warm || frames + 16'd1 == 16'd18) && frames[2:0] + 3'd1 == 3'd2) begin
                        first_slot <= frames[3:0] + 4'd1;
                        sum_count <= '0;
                        log_sum <= '0;
                        state <= SUM;
                    end else state <= IDLE;
                end
            end
            SUM: begin
                // Address sum_count is read this cycle; the previous one arrives now.
                if (sum_count != 0) log_sum <= log_sum + {8'd0, ring_rdata};
                sum_count <= sum_count + 1'b1;
                if (sum_count == 9'd256) begin
                    mean <= 16'((log_sum + {8'd0, ring_rdata}) >> 8);
                    win_g <= 1'b0;
                    win_k <= '0;
                    win_l <= '0;
                    state <= WIN_RD;
                end
            end
            WIN_RD: state <= WIN_CAP;
            WIN_CAP: begin
                word[win_l*LW +: LW] <= LW'(centred);
                win_l <= win_l + 1'b1;
                state <= state_t'((win_l == 3'd7) ? WIN_WRITE : WIN_RD);
            end
            WIN_WRITE: begin
                a_we <= 1'b1;
                a_waddr <= F_BASE + {4'd0, win_g, win_k};
                a_wdata <= word;
                win_k <= win_k + 1'b1;
                if (win_k == 4'd15) win_g <= 1'b1;
                state <= state_t'((win_g && win_k == 4'd15) ? WIN_FLUSH : WIN_RD);
            end
            WIN_FLUSH: state <= WAIT;   // the last registered write lands this cycle
            WAIT: if (resume) state <= IDLE;
            default: state <= IDLE;
        endcase
    end
endmodule
