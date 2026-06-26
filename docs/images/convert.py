import rasterio
import numpy as np
from PIL import Image

def tiff_to_png(tiff_path, png_output_path):
    with rasterio.open(tiff_path) as src:
        # Lese die ersten drei Bänder (RGB)
        # Hinweis: Wenn dein Bild nur 1 Band hat, musst du es entsprechend anpassen
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        
        # Erstelle ein RGB-Array (H, W, 3)
        rgb = np.dstack((r, g, b))
        
        # --- WICHTIG: Skalierung ---
        # NoData-Werte (z.B. -9999) ausschließen
        valid_mask = (r != -9999.0) # Oder dein spezifischer NoData-Wert
        
        # 2% / 98% Perzentil-Stretch für besseren Kontrast
        p2, p98 = np.percentile(rgb[valid_mask], (2, 98))
        rgb_stretched = np.clip((rgb - p2) / (p98 - p2), 0, 1)
        
        # Umwandlung in 8-Bit (0-255)
        rgb_8bit = (rgb_stretched * 255).astype(np.uint8)
        
        # Hintergrund füllen (NoData = Schwarz)
        rgb_8bit[~valid_mask] = 0
        
        # Speichern als PNG
        img = Image.fromarray(rgb_8bit)
        img.save(png_output_path)
        print(f"✅ PNG gespeichert unter: {png_output_path}")

# Anwendung
path = "/net/home/sloeblein/Patras_Wildfire/Data/0_Original_Images/Burned_Orfanos260222/260222_True_Ortho.tif"
target_path = "/net/home/sloeblein/Patras_Wildfire/docs/images/260222_original.png"
tiff_to_png(path, target_path)