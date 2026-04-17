import os
from step52_admin_upload import create_production_app

app = create_production_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=False)
