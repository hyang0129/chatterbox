#!/usr/bin/env bash
# Generate a self-signed TLS certificate for local HTTPS development.
#
# Usage:
#   ./scripts/generate-cert.sh [OUTPUT_DIR]
#
# Defaults to ./certs/ if OUTPUT_DIR is not provided.
# Skips generation if cert already exists (idempotent).

set -euo pipefail

CERT_DIR="${1:-./certs}"
CERT_FILE="$CERT_DIR/localhost.pem"
KEY_FILE="$CERT_DIR/localhost-key.pem"

if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "Certs already exist at $CERT_DIR — skipping generation."
    exit 0
fi

mkdir -p "$CERT_DIR"

echo "Generating self-signed certificate in $CERT_DIR ..."

openssl req -x509 \
    -newkey rsa:2048 \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -sha256 \
    -days 365 \
    -nodes \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1"

chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo "Certificate: $CERT_FILE"
echo "Key:         $KEY_FILE"
echo ""
echo "Usage:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000 \\"
echo "      --ssl-keyfile $KEY_FILE --ssl-certfile $CERT_FILE"
echo ""
echo "To bypass in curl:  curl -k https://localhost:8000/health"
