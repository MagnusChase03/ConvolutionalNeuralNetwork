all: run

run: build
	cargo run --release

build:
	cargo build --release
	strip target/release/cnn

clean:
	cargo clean