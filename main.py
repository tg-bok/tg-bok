import os
from step67_full_concentrated_fix import create_production_app

app = create_production_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=False)
