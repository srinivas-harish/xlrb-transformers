# Makefile (repo root)
.PHONY: dev api worker redis lint test
dev:
	$(MAKE) -C project dev
api:
	$(MAKE) -C project api
worker:
	$(MAKE) -C project worker
redis:
	$(MAKE) -C project redis
