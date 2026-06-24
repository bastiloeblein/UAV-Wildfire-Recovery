import otbApplication as otb
import os
os.environ["OTB_LOGGER_LEVEL"] = "INFO"

# This code is adapted from: https://github.com/pajevicnina/inspire1-seg

def main():
    print("--- Starting OTB LSMS Segmentation ---")

    # 1. DEFINE PATHS
    folder_path = "../../Data/6_Cropped_Images"
    input_image_name = "250821_resampled_cropped.tif"
    folder_out = "../../Data/7_OBIA_Segmentation"
    os.makedirs(folder_out, exist_ok=True) 
    
    # Radius used in paper
    range_radius = 3

    # test_image_name = "test_patch_250821.tif"
    # out_test = os.path.join(folder_out, test_image_name)

    # app0 = otb.Registry.CreateApplication("ExtractROI")
    # app0.SetParameterString("in", os.path.join(folder_path, input_image_name))
    # app0.SetParameterString("out", out_test)
    
    # # Define starting pixel (x, y) and size of the box (width, height in pixels)
    # # Hint: If the output is completely black, you hit the NoData border. 
    # # Just increase startx and starty to move deeper into the center of the image.
    # app0.SetParameterInt("startx", 4000) 
    # app0.SetParameterInt("starty", 4000)
    # app0.SetParameterInt("sizex", 2000)
    # app0.SetParameterInt("sizey", 2000)
    
    # app0.ExecuteAndWriteOutput()

    # # CRITICAL TRICK: Overwrite the input_image_name variable here!
    # input_image_name = test_image_name
    
    # 2. STEP 1: MEAN SHIFT SEGMENTATION
    print(f"Step 1: Segmentation (Range Radius: {range_radius})...")
    full_image_path = os.path.join(folder_path, input_image_name)
    segmentation_image_path = full_image_path + "?&channels=1,2,3,4"
    
    app1 = otb.Registry.CreateApplication("LSMSSegmentation")
    app1.SetParameterString("in", segmentation_image_path)
    
    out_seg = os.path.join(folder_out, f'Segmented_r{int(range_radius)}.tif')
    app1.SetParameterString("out", out_seg)
    app1.SetParameterFloat("ranger", range_radius)  
    # app1.SetParameterFloat("spatialr", 25.0) # Optionally: Spatial Radius
    # app1.SetParameterInt("ram", 8192)
    app1.ExecuteAndWriteOutput()

    # 3. STEP 2: MERGE SMALL REGIONS
    print("Step 2: Merge small regions...")
    app2 = otb.Registry.CreateApplication("LSMSSmallRegionsMerging")
    app2.SetParameterString("in", os.path.join(folder_path, input_image_name))
    app2.SetParameterString("inseg", out_seg)
    
    out_merged = os.path.join(folder_out, f'Small_regions_merged_r{int(range_radius)}.tif')
    app2.SetParameterString("out", out_merged)
    # app2.SetParameterInt("minsize", 800) # Optional: Minimum pixel size for object 
    # app2.SetParameterInt("ram", 8192)
    app2.ExecuteAndWriteOutput()

    # 4. STEP 3: VECTORIZATION (Raster to Polygons)
    print("Step 3: Vectorization to Shapefile...")
    app3 = otb.Registry.CreateApplication("LSMSVectorization")
    app3.SetParameterString("in", os.path.join(folder_path, input_image_name))
    app3.SetParameterString("inseg", out_merged)
    
    out_shp = os.path.join(folder_out, f'Vectorized_r{range_radius}.shp')
    app3.SetParameterString("out", out_shp)
    app3.ExecuteAndWriteOutput()

    print(f"\n--- Done! ---")

if __name__ == "__main__":
    # Do this in console: source /net/home/sloeblein/otb-9.1.0/otbenv.profile 
    main()