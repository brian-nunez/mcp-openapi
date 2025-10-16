all:
	@python server.py

vstart:
	uv venv
	@source .venv/bin/activate
	uv pip install -r requirements.txt

vend:
	@deactivate
