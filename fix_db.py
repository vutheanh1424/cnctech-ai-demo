import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_db():
    try:
        conn = sqlite3.connect('cnc_pro_alerts.db')
        c = conn.cursor()
        c.execute("ALTER TABLE scenarios ADD COLUMN factory_id TEXT")
        conn.commit()
        logger.info("Successfully added factory_id column to scenarios table")
        print("Successfully added factory_id column to scenarios table")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_db()