# Example Makefile for Crosstalk
run:
	cargo run -- --task "Draft a project plan for a Rust-based AI" --models gemini-1.5-pro --iterations 5

clean:
	rm -rf .crosstalk_db
	rm -f crosstalk_verbose.log

watch:
	tail -f crosstalk_verbose.log