// Board-independent accelerator top: UART framing, commands and memories.
//
// Request payload:  op, arg, data...      Response payload: op, status, data...
//   01 ECHO   arg, up to 2048 bytes       -> 01 00 + the same bytes
//   02 LOAD   slot (layer*6 + q,k,v,o,w1,w2; 12 embed; 13 output),
//             lsh, then int8 W[k][c] row-major (64x64; 16x64; 64x16)
//                                          -> 02 00 (01 on bad length/slot)
//   03 ATTN   layer, 16x64 int16 LE X      -> 03 00 + 16x64 int16 LE attention
//   04 INFO                                -> 04 00 'T' 'P' 'U' version cycles[4]
//   05 INFER  0, 16x16 int16 LE features   -> 05 00 class cycles[4] + 16x16 int16 LE logits
//   06 LOADPOS 0, 16x64 int16 LE position  -> 06 00
//   bad checksum                           -> EE 01
// cycles is the core's compute time for the last ATTN/INFER, in clocks.
// Weights, shifts and the position table start from the baked files, so the
// board runs the model from power-up; LOAD/LOADPOS replace them until reset.
// LOAD writes as bytes arrive, so a LOAD answered with EE must be resent.
// The host waits for each response before sending the next request.
module tpu_top #(
    parameter integer CLKS_PER_BIT = 234,
    parameter integer TIMEOUT_CLKS = 2_700_000,
    parameter WEIGHTS_FILE = "weights/tpu_weights.mem",
    parameter LSH_FILE = "weights/tpu_lsh.mem",
    parameter POSITION_FILE = "weights/tpu_position.mem"
) (
    input  logic clk,
    input  logic rst,
    input  logic uart_rx_pin,
    output logic uart_tx_pin,
    output logic busy,
    output logic computing,
    output logic rx_activity,
    output logic error_seen
);
    localparam integer MAX_PAYLOAD = 4099;
    localparam logic [7:0] OP_ECHO = 8'h01, OP_LOAD = 8'h02, OP_ATTN = 8'h03, OP_INFO = 8'h04,
                           OP_INFER = 8'h05, OP_LOADPOS = 8'h06, OP_ERROR = 8'hEE;
    localparam logic [7:0] VERSION = 8'h02;
    localparam logic [2:0] SRC_NONE = 3'd0, SRC_RX = 3'd1, SRC_O = 3'd2, SRC_INFO = 3'd3,
                           SRC_INFER = 3'd4;
    localparam logic [1:0] DST_X = 2'd0, DST_F = 2'd1, DST_POS = 2'd2;

    // UART and framing.
    logic [7:0] rx_byte, payload_byte;
    logic rx_valid, payload_valid, packet_start, packet_valid, packet_error;
    logic [12:0] payload_length;
    logic tx_start, tx_busy;
    logic [7:0] tx_data;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) rx(
        .clk(clk), .rst(rst), .rx(uart_rx_pin), .data(rx_byte), .valid(rx_valid));
    packet_parser #(.MAX_PAYLOAD(MAX_PAYLOAD), .TIMEOUT_CLKS(TIMEOUT_CLKS)) parser(
        .clk(clk), .rst(rst), .byte_in(rx_byte), .byte_valid(rx_valid),
        .payload_data(payload_byte), .payload_valid(payload_valid),
        .packet_start(packet_start), .packet_valid(packet_valid),
        .packet_error(packet_error), .payload_length(payload_length));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) tx(
        .clk(clk), .rst(rst), .start(tx_start), .data(tx_data),
        .tx(uart_tx_pin), .busy(tx_busy));

    // Memories.
    logic a_we;
    logic [8:0] a_waddr, a_raddr;
    logic [143:0] a_wdata, a_rdata;
    sdp_ram #(.DEPTH(512), .WIDTH(144)) amem(
        .clk(clk), .we(a_we), .waddr(a_waddr), .wdata(a_wdata),
        .raddr(a_raddr), .rdata(a_rdata));

    // Weights: 12 layer slots of 1024 words, then embed and output (256 each).
    logic w_we;
    logic [13:0] w_waddr, w_raddr;
    logic [31:0] w_wdata, w_rdata;
    logic [31:0] wmem [0:12799];
    initial $readmemh(WEIGHTS_FILE, wmem);
    always_ff @(posedge clk) begin
        if (w_we) wmem[w_waddr] <= w_wdata;
        w_rdata <= wmem[w_raddr];
    end

    logic [3:0] w_slot;
    logic [3:0] lsh_mem [0:15];
    initial $readmemh(LSH_FILE, lsh_mem);
    logic lsh_we;
    logic [3:0] lsh_slot;
    logic [2:0] lsh_value;
    always_ff @(posedge clk)
        if (lsh_we) lsh_mem[lsh_slot] <= {1'b0, lsh_value};

    logic pos_we;
    logic [6:0] pos_waddr, pos_raddr;
    logic [127:0] pos_wdata, pos_rdata;
    logic [127:0] pos_mem [0:127];
    initial $readmemh(POSITION_FILE, pos_mem);
    always_ff @(posedge clk) begin
        if (pos_we) pos_mem[pos_waddr] <= pos_wdata;
        pos_rdata <= pos_mem[pos_raddr];
    end

    logic rx_we;
    logic [10:0] rx_waddr, rx_raddr;
    logic [7:0] rx_wdata, rx_rdata;
    sdp_ram #(.DEPTH(2048), .WIDTH(8)) rxbuf(
        .clk(clk), .we(rx_we), .waddr(rx_waddr), .wdata(rx_wdata),
        .raddr(rx_raddr), .rdata(rx_rdata));

    // Model core.
    logic core_start, core_prog, core_busy, core_done, layer;
    logic [3:0] core_class;
    logic [31:0] core_cycles;
    logic core_a_we;
    logic [8:0] core_a_raddr, core_a_waddr;
    logic [143:0] core_a_wdata;
    tpu_core core(
        .clk(clk), .rst(rst), .start(core_start), .prog(core_prog), .layer(layer),
        .busy(core_busy), .done(core_done), .class_out(core_class), .cycles(core_cycles),
        .a_raddr(core_a_raddr), .a_rdata(a_rdata), .a_we(core_a_we),
        .a_waddr(core_a_waddr), .a_wdata(core_a_wdata),
        .w_raddr(w_raddr), .w_rdata(w_rdata), .w_slot(w_slot), .w_lsh(lsh_mem[w_slot][2:0]),
        .pos_raddr(pos_raddr), .pos_rdata(pos_rdata));

    typedef enum logic [1:0] {T_IDLE, T_REPACK, T_RUN, T_RESP} top_state_t;
    typedef enum logic [1:0] {R_FETCH, R_SEND, R_WAIT} resp_state_t;
    top_state_t state;
    resp_state_t rstate;

    // Request capture.
    logic [12:0] byte_idx;
    // A 64x64 LOAD carries 4097 data bytes, so data_idx needs 13 bits.
    logic [12:0] data_idx;
    logic [11:0] value_idx;
    logic [9:0] word_idx;
    logic [7:0] opcode, arg;
    logic [23:0] w_word;
    logic [13:0] load_addr;
    assign data_idx = byte_idx - 13'd2;
    assign value_idx = 12'(data_idx - 13'd1);
    assign word_idx = value_idx[11:2];

    // Word value_idx/4 of W[k][c] holds W[k][4cg..4cg+3]; its address is
    // slot base + cg*K + k for a K-row matrix.
    always_comb begin
        case (arg[3:0])
            4'd12: load_addr = 14'd12288 + {6'd0, word_idx[3:0], word_idx[7:4]};
            4'd13: load_addr = 14'd12544 + {6'd0, word_idx[1:0], word_idx[7:2]};
            default: load_addr = {arg[3:0], word_idx[3:0], word_idx[9:4]};
        endcase
    end

    // Response sequencing: pos 0..3 header, 4 op, 5 status, data, checksum.
    logic [7:0] resp_op, resp_status;
    logic [2:0] resp_src;
    logic [11:0] resp_data_len;
    logic [12:0] pos;
    logic [11:0] resp_i;
    logic [8:0] logit_i;
    logic [12:0] resp_payload_len;
    logic [7:0] resp_checksum;
    logic [7:0] resp_byte;
    logic [31:0] last_cycles;
    logic [3:0] last_class;
    assign resp_payload_len = {1'd0, resp_data_len} + 13'd2;
    assign resp_i = 12'(pos - 13'd6);
    assign logit_i = 9'(resp_i - 12'd5);

    always_comb begin
        if (pos == 0) resp_byte = 8'hA5;
        else if (pos == 1) resp_byte = 8'h5A;
        else if (pos == 2) resp_byte = resp_payload_len[7:0];
        else if (pos == 3) resp_byte = {3'd0, resp_payload_len[12:8]};
        else if (pos == 4) resp_byte = resp_op;
        else if (pos == 5) resp_byte = resp_status;
        else if (pos < 13'd6 + {1'd0, resp_data_len}) begin
            case (resp_src)
                SRC_RX: resp_byte = rx_rdata;
                // Byte i of row-major O: row i[10:7] (lane i[9:7]), column i[6:1].
                SRC_O: resp_byte = resp_i[0] ? a_rdata[resp_i[9:7]*18 + 8 +: 8]
                                             : a_rdata[resp_i[9:7]*18 +: 8];
                SRC_INFO: case (resp_i[2:0])
                    3'd0: resp_byte = 8'h54;
                    3'd1: resp_byte = 8'h50;
                    3'd2: resp_byte = 8'h55;
                    3'd3: resp_byte = VERSION;
                    default: resp_byte = last_cycles[(resp_i[1:0] - 2'd0) * 8 +: 8];
                endcase
                // Class, cycles, then logit (t, j) at byte 5 + (t*16 + j)*2 + hi.
                SRC_INFER: begin
                    if (resp_i == 0) resp_byte = {4'd0, last_class};
                    else if (resp_i < 5) resp_byte = last_cycles[(resp_i[1:0] - 2'd1) * 8 +: 8];
                    else resp_byte = logit_i[0] ? a_rdata[logit_i[7:5]*18 + 8 +: 8]
                                                : a_rdata[logit_i[7:5]*18 +: 8];
                end
                default: resp_byte = 8'h00;
            endcase
        end else resp_byte = resp_checksum;
    end

    // Repack: rx bytes of a row-major 16xK int16 matrix -> row-group words.
    // K = 64: rp = {g, k[5:0], l, hi}, byte {g, l, k, hi};  K = 16: rp[8:0] = {g, k[3:0], l, hi}.
    logic [10:0] rp, rp_d;
    logic rp_active, rp_valid;
    logic [1:0] rp_dst;
    logic rp_k16, rp_last, rp_word_done;
    logic [143:0] rp_word, rp_next;
    logic [127:0] rp_pos_word;
    logic rp_g;
    logic [5:0] rp_k;
    assign rp_k16 = rp_dst == DST_F;
    assign rp_last = rp_k16 ? rp[8:0] == 9'h1FF : rp == 11'h7FF;
    assign rp_g = rp_k16 ? rp_d[8] : rp_d[10];
    assign rp_k = rp_k16 ? {2'd0, rp_d[7:4]} : rp_d[9:4];
    assign rp_word_done = rp_valid && rp_d[3:0] == 4'hF;
    always_comb begin
        rp_next = rp_word;
        rp_next[rp_d[3:1]*18 + rp_d[0]*8 +: 8] = rx_rdata;
        if (rp_d[0])
            rp_next[rp_d[3:1]*18 + 16 +: 2] = {2{rx_rdata[7]}};
    end
    genvar pl;
    generate
        for (pl = 0; pl < 8; pl = pl + 1) begin : g_pos_lane
            assign rp_pos_word[pl*16 +: 16] = rp_next[pl*18 +: 16];
        end
    endgenerate
    assign pos_we = rp_word_done && rp_dst == DST_POS;
    assign pos_waddr = {rp_g, rp_k};
    assign pos_wdata = rp_pos_word;

    // Memory port ownership.
    always_comb begin
        if (state == T_REPACK)
            rx_raddr = rp_k16 ? {2'd0, rp[8], rp[3:1], rp[7:4], rp[0]}
                              : {rp[10], rp[3:1], rp[9:4], rp[0]};
        else
            rx_raddr = resp_i[10:0];
        if (state == T_RUN) begin
            a_raddr = core_a_raddr;
            a_we = core_a_we;
            a_waddr = core_a_waddr;
            a_wdata = core_a_wdata;
        end else begin
            // O row r lives in word (r[3], c) of X; logit (t, j) in word 384 + t[3]*16 + j.
            a_raddr = (resp_src == SRC_INFER) ? 9'd384 + {4'd0, logit_i[8], logit_i[4:1]}
                                              : {2'd0, resp_i[10], resp_i[6:1]};
            a_we = rp_word_done && rp_dst != DST_POS;
            a_waddr = rp_k16 ? 9'd448 + {4'd0, rp_g, rp_k[3:0]} : {2'd0, rp_g, rp_k};
            a_wdata = rp_next;
        end
    end

    assign busy = state != T_IDLE;
    assign computing = core_busy;

    always_ff @(posedge clk) begin
        w_we <= 1'b0;
        lsh_we <= 1'b0;
        rx_we <= 1'b0;
        tx_start <= 1'b0;
        core_start <= 1'b0;
        rx_activity <= rx_valid;
        if (rst) begin
            state <= T_IDLE;
            rstate <= R_FETCH;
            byte_idx <= '0;
            opcode <= '0;
            arg <= '0;
            pos <= '0;
            rp <= '0;
            rp_active <= 1'b0;
            rp_valid <= 1'b0;
            rp_dst <= DST_X;
            layer <= 1'b0;
            core_prog <= 1'b0;
            error_seen <= 1'b0;
            last_cycles <= '0;
            last_class <= '0;
        end else begin
            if (packet_start)
                byte_idx <= '0;
            if (payload_valid && state == T_IDLE) begin
                byte_idx <= byte_idx + 1'b1;
                if (byte_idx == 0) opcode <= payload_byte;
                if (byte_idx == 1) arg <= payload_byte;
                if (byte_idx >= 2) begin
                    if (opcode == OP_LOAD) begin
                        if (data_idx == 0) begin
                            lsh_we <= arg < 8'd14;
                            lsh_slot <= arg[3:0];
                            lsh_value <= payload_byte[2:0];
                        end else if (value_idx[1:0] != 2'd3) begin
                            w_word[value_idx[1:0]*8 +: 8] <= payload_byte;
                        end else begin
                            w_we <= arg < 8'd14;
                            w_waddr <= load_addr;
                            w_wdata <= {payload_byte, w_word};
                        end
                    end else if (data_idx < 12'd2048) begin
                        rx_we <= 1'b1;
                        rx_waddr <= data_idx[10:0];
                        rx_wdata <= payload_byte;
                    end
                end
            end

            case (state)
                T_IDLE: begin
                    pos <= '0;
                    resp_checksum <= '0;
                    rstate <= R_FETCH;
                    resp_src <= SRC_NONE;
                    resp_data_len <= '0;
                    resp_status <= 8'h00;
                    resp_op <= opcode;
                    if (packet_error) begin
                        error_seen <= 1'b1;
                        resp_op <= OP_ERROR;
                        resp_status <= 8'h01;
                        state <= T_RESP;
                    end else if (packet_valid && payload_length == 0) begin
                        resp_op <= OP_ERROR;
                        resp_status <= 8'h02;
                        state <= T_RESP;
                    end else if (packet_valid) begin
                        state <= T_RESP;
                        case (opcode)
                            OP_ECHO: if (payload_length <= 13'd2050) begin
                                resp_src <= SRC_RX;
                                resp_data_len <= (payload_length > 13'd2) ?
                                                 12'(payload_length - 13'd2) : 12'd0;
                            end else resp_status <= 8'h01;
                            OP_LOAD: if (arg > 8'd13 || payload_length !=
                                         ((arg >= 8'd12) ? 13'd1027 : 13'd4099))
                                resp_status <= 8'h01;
                            OP_ATTN, OP_LOADPOS: if (payload_length == 13'd2050 &&
                                                     (opcode == OP_LOADPOS || arg < 8'd2)) begin
                                layer <= arg[0];
                                core_prog <= 1'b0;
                                rp <= '0;
                                rp_dst <= (opcode == OP_LOADPOS) ? DST_POS : DST_X;
                                rp_active <= 1'b1;
                                state <= T_REPACK;
                            end else resp_status <= 8'h01;
                            OP_INFER: if (payload_length == 13'd514) begin
                                core_prog <= 1'b1;
                                rp <= '0;
                                rp_dst <= DST_F;
                                rp_active <= 1'b1;
                                state <= T_REPACK;
                            end else resp_status <= 8'h01;
                            OP_INFO: begin
                                resp_src <= SRC_INFO;
                                resp_data_len <= 12'd8;
                            end
                            default: resp_status <= 8'h02;
                        endcase
                    end
                end
                T_REPACK: begin
                    rp_valid <= rp_active;
                    rp_d <= rp;
                    if (rp_valid) rp_word <= rp_next;
                    if (rp_active) begin
                        rp <= rp + 1'b1;
                        if (rp_last) rp_active <= 1'b0;
                    end else if (!rp_valid) begin
                        if (opcode == OP_LOADPOS) state <= T_RESP;
                        else begin
                            core_start <= 1'b1;
                            state <= T_RUN;
                        end
                    end
                end
                T_RUN: if (core_done) begin
                    last_cycles <= core_cycles;
                    last_class <= core_class;
                    resp_src <= core_prog ? SRC_INFER : SRC_O;
                    resp_data_len <= core_prog ? 12'd517 : 12'd2048;
                    state <= T_RESP;
                end
                T_RESP: case (rstate)
                    // Addresses derive from pos; give the BSRAM one cycle.
                    R_FETCH: rstate <= R_SEND;
                    R_SEND: if (!tx_busy) begin
                        tx_data <= resp_byte;
                        tx_start <= 1'b1;
                        if (pos >= 13'd4) resp_checksum <= resp_checksum ^ resp_byte;
                        rstate <= R_WAIT;
                    end
                    R_WAIT: if (tx_busy) begin
                        if (pos == 13'd6 + {1'd0, resp_data_len}) begin
                            state <= T_IDLE;
                        end else begin
                            pos <= pos + 1'b1;
                            rstate <= R_FETCH;
                        end
                    end
                    default: rstate <= R_FETCH;
                endcase
                default: state <= T_IDLE;
            endcase
        end
    end
endmodule
