#!/bin/bash
set -e

# 0. Install System Dependencies
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-pip nginx

# 1. Setup Python Environment
cd /home/filip/studio-website
rm -rf venv # Clean up failed install
python3 -m venv venv
source venv/bin/activate
pip install flask jax flax numpy optax distrax

# 2. Create Systemd Service
cat <<EOF | sudo tee /etc/systemd/system/jambweb.service
[Unit]
Description=JambRL Web App
After=network.target

[Service]
User=filip
WorkingDirectory=/home/filip/studio-website
Environment="PATH=/home/filip/studio-website/venv/bin"
ExecStart=/home/filip/studio-website/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jambweb
sudo systemctl restart jambweb

# 3. Update Nginx Config
# We use sed to replace the location block. This is fragile but efficient.
# Current block: "location / { try_files $uri $uri/ =404; }"
# We want to replace it with proxy_pass.

# Easier: Overwrite the file with a known good config block if strict.
# But messing with SSL cert paths in full overwrite is dangerous.
# Let's perform a careful Sed replacement.
# Replace "root /home/filip/studio-website;" with comment or line deletion (since proxy doesn't need root, but keeping it doesn't hurt)
# Replace "try_files $uri $uri/ =404;" with "proxy_pass http://127.0.0.1:5000;"

CONF="/etc/nginx/sites-available/essencedistillery.conf"

# Backup
sudo cp $CONF ${CONF}.bak

# Replace the location block logic
# We'll just append the proxy block and comment out the old one? No.
# Let's construct the file content manually, preserving the cert paths.
# Actually, the user's config was simple.

# Reading cert paths from previous `cat` output in memory:
# ssl_certificate /etc/letsencrypt/live/essencedistillery.soon.it/fullchain.pem;
# ssl_certificate_key /etc/letsencrypt/live/essencedistillery.soon.it/privkey.pem;

cat <<EOF | sudo tee $CONF
server {
    listen 80;
    listen [::]:80;
    server_name essencedistillery.soon.it;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name essencedistillery.soon.it;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/essencedistillery.soon.it/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/essencedistillery.soon.it/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    access_log /var/log/nginx/essencedistillery.access.log;
    error_log /var/log/nginx/essencedistillery.error.log;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }

    location /static {
        alias /home/filip/studio-website/static;
    }
}
EOF

sudo nginx -t
sudo systemctl reload nginx

echo "✅ Deployment Complete!"
