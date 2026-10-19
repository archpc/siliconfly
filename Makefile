.PHONY: run fly build test serve clean

# Default: headless experimental lab server (HTTP bridge on 127.0.0.1:8080).
# The GUI desktop fly is still available via `make fly` or ./SiliconFly.
run: serve

# Launch the GUI desktop fly; quit from the menu-bar 🪰.
fly:
	@./SiliconFly

build:
	@export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && ./build.sh

# GPU/CPU bit-exactness + determinism validation (~10 s).
test:
	@./SiliconFly --gpucheck

# Headless lab bridge (spawns ./SiliconFly --serve under the hood).
serve:
	@cd .. && python3 lab.py

clean:
	@rm -f SiliconFly
