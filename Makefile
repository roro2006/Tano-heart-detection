PYTHON ?= python3
# Point TOOLS at an OSS CAD Suite install, or leave empty to use PATH.
TOOLS ?=
BIN := $(if $(TOOLS),$(TOOLS)/bin/,)
IVERILOG ?= $(BIN)iverilog
VVP ?= $(BIN)vvp
YOSYS ?= $(BIN)yosys
NEXTPNR ?= $(BIN)nextpnr-himbaechel
GOWIN_PACK ?= $(BIN)gowin_pack
LOADER ?= $(BIN)openFPGALoader

DEVICE := GW2AR-LV18QN88C8/I7
FAMILY := GW2A-18C
BUILD := build
CST := constraints/tang_nano_20k.cst

CORE_RTL := rtl/pe.sv rtl/systolic_8x4.sv rtl/exp_rom.sv rtl/softmax_unit.sv rtl/ln_unit.sv \
            rtl/sdp_ram.sv rtl/tpu_core.sv rtl/uart_rx.sv rtl/uart_tx.sv \
            rtl/packet_parser.sv rtl/tpu_top.sv
TOP_RTL := $(CORE_RTL) rtl/tang_nano_20k_top.sv

.PHONY: test generate export dataset sim sim-systolic sim-uart sim-ln sim-top synth pnr bitstream prog flash clean

test:
	PYTHONPATH=python $(PYTHON) -m unittest discover -s python -p 'test_*.py'

generate:
	$(PYTHON) python/generate_artifacts.py

# Convert weights/ (model_export format) to the on-chip model and the BSRAM
# images the bitstream bakes in; rebuild the bitstream afterwards.
export:
	$(PYTHON) python/tpu_model.py

dataset:
	$(PYTHON) python/dataset.py

$(BUILD):
	mkdir -p $(BUILD)

sim: sim-systolic sim-uart sim-ln sim-top

sim-systolic: | $(BUILD)
	$(IVERILOG) -g2012 -o $(BUILD)/tb_systolic sim/tb_systolic_8x4.sv rtl/pe.sv rtl/systolic_8x4.sv
	$(VVP) -n $(BUILD)/tb_systolic

sim-uart: | $(BUILD)
	$(IVERILOG) -g2012 -o $(BUILD)/tb_uart sim/tb_uart_echo.sv rtl/uart_echo_top.sv rtl/uart_rx.sv rtl/uart_tx.sv
	$(VVP) -n $(BUILD)/tb_uart

sim-ln: | $(BUILD)
	$(PYTHON) python/sim_vectors.py --ln
	$(IVERILOG) -g2012 -o $(BUILD)/tb_ln sim/tb_ln_unit.sv rtl/ln_unit.sv
	$(VVP) -n $(BUILD)/tb_ln

# Full UART -> model -> UART check against the Python oracle (~2 minutes).
sim-top: | $(BUILD)
	$(PYTHON) python/sim_vectors.py
	$(IVERILOG) -g2012 -o $(BUILD)/tb_tpu_top sim/tb_tpu_top.sv $(CORE_RTL)
	$(VVP) -n $(BUILD)/tb_tpu_top

$(BUILD)/tpu.json: $(TOP_RTL) | $(BUILD)
	$(YOSYS) -l $(BUILD)/yosys.log -q -p "read_verilog -sv $(TOP_RTL); synth_gowin -family gw2a -top tang_nano_20k_top -json $@"

$(BUILD)/tpu_pnr.json: $(BUILD)/tpu.json $(CST)
	$(NEXTPNR) --json $< --write $@ --device $(DEVICE) --vopt family=$(FAMILY) \
		--vopt cst=$(CST) --freq 27 --log $(BUILD)/nextpnr.log -q

$(BUILD)/tpu.fs: $(BUILD)/tpu_pnr.json
	$(GOWIN_PACK) -d $(FAMILY) -o $@ $<

synth: $(BUILD)/tpu.json
pnr: $(BUILD)/tpu_pnr.json
bitstream: $(BUILD)/tpu.fs

# Load into SRAM (lost on power cycle) or write the on-board flash.
prog: $(BUILD)/tpu.fs
	$(LOADER) -b tangnano20k $<

flash: $(BUILD)/tpu.fs
	$(LOADER) -b tangnano20k -f $<

clean:
	rm -rf $(BUILD)
