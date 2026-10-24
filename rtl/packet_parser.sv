// Frame parser: A5 5A, little-endian length, payload, XOR checksum.
// packet_start pulses once the length is accepted; payload bytes then stream
// out before the checksum is known, so consumers must honour packet_error.
// A gap longer than TIMEOUT_CLKS inside a frame drops back to hunting for A5.
module packet_parser #(
    parameter integer MAX_PAYLOAD = 2048,
    parameter integer TIMEOUT_CLKS = 2_700_000
) (
    input  logic clk,
    input  logic rst,
    input  logic [7:0] byte_in,
    input  logic byte_valid,
    output logic [7:0] payload_data,
    output logic payload_valid,
    output logic packet_start,
    output logic packet_valid,
    output logic packet_error,
    output logic [$clog2(MAX_PAYLOAD+1)-1:0] payload_length
);
    typedef enum logic [2:0] {WAIT_A5, WAIT_5A, LEN_LO, LEN_HI, PAYLOAD, CHECKSUM} state_t;
    state_t state;
    logic [15:0] expected_length, index;
    logic [7:0] checksum;
    logic [$clog2(TIMEOUT_CLKS+1)-1:0] idle;

    always_ff @(posedge clk) begin
        payload_valid <= 1'b0;
        packet_start <= 1'b0;
        packet_valid <= 1'b0;
        packet_error <= 1'b0;
        if (rst) begin
            state <= WAIT_A5;
            expected_length <= '0;
            index <= '0;
            checksum <= '0;
            payload_length <= '0;
            idle <= '0;
        end else if (byte_valid) begin
            idle <= '0;
            case (state)
                WAIT_A5: state <= state_t'((byte_in == 8'hA5) ? WAIT_5A : WAIT_A5);
                WAIT_5A: state <= state_t'((byte_in == 8'h5A) ? LEN_LO :
                                  (byte_in == 8'hA5) ? WAIT_5A : WAIT_A5);
                LEN_LO: begin expected_length[7:0] <= byte_in; state <= LEN_HI; end
                LEN_HI: begin
                    expected_length[15:8] <= byte_in;
                    index <= 0;
                    checksum <= 0;
                    if ({byte_in, expected_length[7:0]} > MAX_PAYLOAD)
                        state <= WAIT_A5;
                    else begin
                        packet_start <= 1'b1;
                        state <= state_t'(({byte_in, expected_length[7:0]} == 0) ? CHECKSUM : PAYLOAD);
                    end
                end
                PAYLOAD: begin
                    payload_data <= byte_in;
                    payload_valid <= 1'b1;
                    checksum <= checksum ^ byte_in;
                    index <= index + 1'b1;
                    if (index + 1 >= expected_length)
                        state <= CHECKSUM;
                end
                CHECKSUM: begin
                    payload_length <= expected_length[$bits(payload_length)-1:0];
                    if (byte_in == checksum) packet_valid <= 1'b1;
                    else packet_error <= 1'b1;
                    state <= WAIT_A5;
                end
                default: state <= WAIT_A5;
            endcase
        end else if (state != WAIT_A5) begin
            if (idle == TIMEOUT_CLKS) begin
                idle <= '0;
                state <= WAIT_A5;
            end else idle <= idle + 1'b1;
        end
    end
endmodule
