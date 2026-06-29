import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from arosics import COREG_LOCAL, COREG
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

NO_DATA_VALUE = 0.0

def standardize_and_align(file_path, nodatavalue=NO_DATA_VALUE):
    """
    Reads any raster, forces it to float32, aligns NoData to -9999.0,
    and returns the path to the newly saved, clean file.
    """
    clean_path = file_path.replace(".tif", "_aligned.tif")
    
    with rasterio.open(file_path) as src:
        meta = src.meta.copy()
        # 1. Force to float32 so we can safely store  0.0
        data = src.read().astype(np.float32)
        old_nodata = src.nodata
        
        # 2. Catch officially registered NoData values (e.g., 0.0 or -10000.0)
        if old_nodata is not None:
            data[data == old_nodata] = nodatavalue
            
        # 3. Catch unregistered anomalies (like your 'None' RGBs that have 0.0 borders)
        data[data == -10000.0] = nodatavalue
        data[data == 0.0] =nodatavalue  # Assumes 0.0 is always the border/background
        
        # 4. Update metadata to the new universal standard
        meta.update({
            "nodata": nodatavalue,
            "dtype": "float32"
        })
        
    with rasterio.open(clean_path, "w", **meta) as dst:
        dst.write(data)
        
    return clean_path


def stack_rgb_and_dsm(rgb_path, dsm_path, target_crs, out_path, nodatavalue=NO_DATA_VALUE):

    with rasterio.open(rgb_path) as rgb_src, rasterio.open(dsm_path) as dsm_src:

        # 1. Read bands 
        rgb_data = rgb_src.read().astype(np.float32)   # Shape: (C, H, W)
        dsm_data = dsm_src.read(1).astype(np.float32) # Shape: (1, H, W)

        # 2. If RGB has more than 3 channels (due to alpha channel, remove it)
        if rgb_data.shape[0] > 3:
            rgb_data = rgb_data[:3]

        # 3. Add Band channel dimension to DSM 
        dsm_data = np.expand_dims(dsm_data, axis=0)  # Shape: (1, H, W)

        # 4. Stack RGB and DSM along the Band-Axis
        stacked_data = np.concatenate([rgb_data, dsm_data], axis=0)  # Shape: (4, H, W)

        # 5. Copy meta data and update channel number, data type, CRS, AND NoData!        
        meta = rgb_src.meta.copy()
        meta.update({
            "count": 4,  
            "dtype": "float32",  
            "crs": target_crs,
            "nodata": nodatavalue  
        })

        # 6. Save 4 channel geotiff
        with rasterio.open(out_path, "w", **meta) as dest:
            dest.write(stacked_data)

    print("Successfully stacked RGB and DSM!")

def calculate_intersection_and_master(stacked_dir, all_image_paths, target_res):
    """
    Calculates common intersection of all images and creates a
    Master-Grid based on that with the target resolution.
    """
    lefts, bottoms, rights, tops = [], [], [], []
    master_crs = None
    
    # 1. Collect bounds of all images
    for img in all_image_paths:
        if not img.endswith('.tif'): 
            continue
            
        img_path = os.path.join(stacked_dir, img)
        with rasterio.open(img_path) as src:
            lefts.append(src.bounds.left)
            bottoms.append(src.bounds.bottom)
            rights.append(src.bounds.right)
            tops.append(src.bounds.top)
            if master_crs is None:
                master_crs = src.crs

    # 2. Get intersection
    common_left = max(lefts)
    common_bottom = max(bottoms)
    common_right = min(rights)
    common_top = min(tops)
    
    # 3. Master Transform and calculate shape based on intersection
    master_transform = from_origin(
        common_left, 
        common_top, 
        target_res, 
        target_res
    )
    
    master_width = int((common_right - common_left) / target_res)
    master_height = int((common_top - common_bottom) / target_res)
    
    return master_transform, master_width, master_height, master_crs


def resample_to_master_grid(image_path, out_path, master_transform, master_width, master_height, master_crs):
    """
    Forces image to Master-Grid.
    """
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        
        profile.update({
            "crs": master_crs,
            "transform": master_transform,
            "width": master_width,
            "height": master_height,
            "nodata": NO_DATA_VALUE,
            "compress": "lzw"
        })

        # Create empty array for all channels (src.count)
        data = np.full((src.count, master_height, master_width), NO_DATA_VALUE, dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, tuple(range(1, src.count + 1))),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master_transform,
            dst_crs=master_crs,
            src_nodata=NO_DATA_VALUE, 
            dst_nodata=NO_DATA_VALUE,
            resampling=Resampling.cubic
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

        print("===================================")
        print(f"FILE: {os.path.basename(image_path)}")
        print(f"Original Shape:      {(src.height, src.width)}")
        print(f"New Shape:           {(master_height, master_width)}")

def stretch_img(arr, nodata_val):
    """Helper function for better contrast when plotting."""
    arr_masked = np.ma.masked_equal(arr, nodata_val)
    vmin = np.nanpercentile(arr_masked.compressed(), 2)
    vmax = np.nanpercentile(arr_masked.compressed(), 98)
    return rescale_intensity(arr_masked, in_range=(vmin, vmax), out_range='int8')


def run_global_coregistration(ref_image_path, target_image_paths, out_dir_global, kwargs_global):
    """
    Executes global coregistration.
    
    :param ref_image_path: Path to master/reference image
    :param target_image_paths: Listof paths of the target images (the ones that should be coregistrated to the ref image)
    :param out_dir_global: Output folder
    :param kwargs_global: Dictionary with AROSICS params
    """
    
    for file in target_image_paths:

        basename = os.path.basename(file).replace(".tif", "").split("_")[0]

        # Create folder
        flight_dir = os.path.join(out_dir_global, basename)
        os.makedirs(flight_dir, exist_ok=True)

        # Define filepaths
        out_file = os.path.join(flight_dir, f"{basename}_coreg.tif")
        footprint_file = os.path.join(flight_dir, f"{basename}_footprint.html")
        json_file = os.path.join(flight_dir, f"{basename}_global_report.json") 
        window_plot_file = os.path.join(flight_dir, f"{basename}_matching_window_comparison.png")
        scps_plot_file = os.path.join(flight_dir, f"{basename}_cross_power_spectrum.png")

        print(f"\n=========================================")
        print(f"--- Processing: {basename} ---")
        print(f"=========================================")
    
        try:
            CRL = COREG(
                ref_image_path,
                file,
                path_out=out_file,
                **kwargs_global
            )
            status = CRL.calculate_spatial_shifts()
            if status == 'fail': 
                raise Exception("Normal matching failed, triggering fallback.")
            
        except Exception as e:
            print(f"Standard-Matching failed. Use new coordinates!")
            fallback_kwargs = kwargs_global.copy()
            fallback_kwargs["wp"] = (557178.18, 4219534.37)
            CRL = COREG(
                ref_image_path,
                file,
                path_out=out_file,
                **fallback_kwargs
            )
            status = CRL.calculate_spatial_shifts()

        print(f"Status: {status}")

        if status == 'success':
            CRL.correct_shifts()
            print("Finished correction")
    
            report = {
                "Status": "Success",
                "Shift_X_meters": float(CRL.x_shift_map),
                "Shift_Y_meters": float(CRL.y_shift_map),
                "Vector_Length_meters": float(CRL.vec_length_map),
                "Vector_Angle_degrees": float(CRL.vec_angle_deg),
                "Confidence_Reliability_percent": float(CRL.shift_reliability),
                "SSIM_Before": float(CRL.ssim_orig),
                "SSIM_After": float(CRL.ssim_deshifted),
                "SSIM_Improved": bool(CRL.ssim_improved), 
                "Matching_Window_Position_XY": [float(x) for x in CRL.win_pos_XY],
                "Matching_Window_Size": [int(x) for x in CRL.win_size_XY]
            }
    
            print(f"Success! Reliability: {report['Confidence_Reliability_percent']:.1f}%")
            print(f"Shift: {report['Shift_X_meters']:.2f}m X / {report['Shift_Y_meters']:.2f}m Y (In total: {report['Vector_Length_meters']:.2f}m)")
            print(f"SSIM improved: {report['SSIM_Before']:.4f} -> {report['SSIM_After']:.4f}")
    
            # --- PLOT 1: Footprint Map ---
            footprint_map = CRL.show_image_footprints()
            footprint_map.save(footprint_file)
            print(f"Interactive Footprint Map saved to: {footprint_file}")
    
            # --- PLOT 2: Cross Power Spectrum (SCPS)  ---
            CRL.show_cross_power_spectrum(interactive=False) 
            plt.savefig(scps_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SCPS 3D-Plot saved to: {scps_plot_file}")
            
            # --- PLOT 3: Matching-Window ---
            ref_img = CRL.matchWin[:]
            tgt_img_shifted = CRL._get_deshifted_otherWin()[:]
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            axes[0].imshow(stretch_img(ref_img, CRL.matchWin.nodata), cmap='gray')
            axes[0].set_title("Reference Window (Master)")
            axes[0].axis('off')
    
            axes[1].imshow(stretch_img(tgt_img_shifted, CRL.matchWin.nodata), cmap='gray')
            axes[1].set_title(f"Target Window (Shifted by {CRL.vec_length_map:.2f}m)")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(window_plot_file, dpi=300)
            plt.close(fig)
            print(f"Matching Window Comparison saved to: {window_plot_file}")
    
            # --- PLOT 4: Matching Window ---
            CRL.show_matchWin(interactive=False, after_correction=True)
    
        else:
            print("WARNINING: Global Matching failed!")
            report = {
                "Status": "Failed",
                "Errors": [str(e) for e in CRL.tracked_errors]
            }
    
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Report saved: {json_file}")


def run_local_coregistration(ref_image_path, target_image_paths, out_dir_local, kwargs_local):
    """
    Executes local coregistration (gum-sheet transformation) for a list of images.
    
    :param ref_image_path: Path to master/reference image
    :param target_image_paths: List of paths to the globally coregistered target images
    :param out_dir_local: Output folder for the local coregistration results
    :param kwargs_local: Dictionary with AROSICS COREG_LOCAL params
    """
    
    for file in target_image_paths:

        basename = os.path.basename(file).replace(".tif", "").replace("_coreg", "").split("_")[0]

        # Create folder
        flight_dir = os.path.join(out_dir_local, basename)
        os.makedirs(flight_dir, exist_ok=True)
        
        # Define paths
        out_file = os.path.join(flight_dir, f"{basename}_local_coreg.tif")
        csv_file = os.path.join(flight_dir, f"{basename}_tiepoints.csv")
        plot_points_file = os.path.join(flight_dir, f"{basename}_plot_points.png")
        plot_vectors_file = os.path.join(flight_dir, f"{basename}_plot_vectors.png")
        json_file = os.path.join(flight_dir, f"{basename}_report.json")

        print(f"\n=========================================")
        print(f"--- Processing LOCAL: {basename} ---")
        print(f"=========================================")

        CRL = COREG_LOCAL(
            ref_image_path,
            file,
            path_out=out_file,
            **kwargs_local
        )

        try:
            CRL.correct_shifts()

            if getattr(CRL, 'CoRegPoints_table', None) is not None and not CRL.CoRegPoints_table.empty:
                df = CRL.CoRegPoints_table
                
                df.to_csv(csv_file, index=False)
                print(f"Saved Table: {csv_file}")
                
                if 'OUTLIER' in df.columns:
                    valid_points = df[df['OUTLIER'] == False]
                else:
                    valid_points = df

                report = {
                    "Status": "Success",
                    "Total_Points_Calculated": int(len(df)),
                    "Valid_Points_Used": int(len(valid_points)),
                    "Mean_Shift_X_pixels": float(valid_points['X_SHIFT_PX'].mean()) if 'X_SHIFT_PX' in valid_points else 0.0,
                    "Mean_Shift_Y_pixels": float(valid_points['Y_SHIFT_PX'].mean()) if 'Y_SHIFT_PX' in valid_points else 0.0,
                    "Mean_Shift_X_meters": float(valid_points['X_SHIFT_M'].mean()) if 'X_SHIFT_M' in valid_points else 0.0,
                    "Mean_Shift_Y_meters": float(valid_points['Y_SHIFT_M'].mean()) if 'Y_SHIFT_M' in valid_points else 0.0,
                    "Mean_Absolute_Shift_meters": float(valid_points['ABS_SHIFT'].mean()) if 'ABS_SHIFT' in valid_points else 0.0,
                    "Mean_Reliability_percent": float(valid_points['RELIABILITY'].mean()) if 'RELIABILITY' in valid_points else 0.0,
                }

                if 'SSIM_BEFORE' in valid_points.columns and 'SSIM_AFTER' in valid_points.columns:
                    report["Mean_SSIM_Before"] = float(valid_points['SSIM_BEFORE'].mean())
                    report["Mean_SSIM_After"] = float(valid_points['SSIM_AFTER'].mean())

                with open(json_file, 'w') as f:
                    json.dump(report, f, indent=4)
                print(f"Saved Report: {json_file}")
                
                # --- PLOT 1: TIE-POINTS ---
                CRL.view_CoRegPoints(
                    figsize=(10, 10),
                    backgroundIm='ref',
                    savefigPath=plot_points_file,
                    savefigDPI=150       
                )
                plt.show()
                print(f"Saved Tie-Point-Plot: {plot_points_file}")
                
                # --- PLOT 2: VEKTORS ---
                CRL.view_CoRegPoints(
                    figsize=(10, 10),
                    shapes2plot='vectors', 
                    vector_scale=3000,      
                    hide_filtered=True,    
                    backgroundIm='ref',
                    savefigPath=plot_vectors_file,
                    savefigDPI=150
                )
                plt.show()
                print(f"Vector-Plot saved: {plot_vectors_file}")

            else:
                print("WARNUNG: No valid Tie-Points found.")
                with open(json_file, 'w') as f:
                    json.dump({"Status": "Failed", "Reason": "No valid tie points found"}, f, indent=4)

        except Exception as e:
            print(f"Error in local Coregistration of {basename}: {e}")
            with open(json_file, 'w') as f:
                json.dump({"Status": "Error", "Reason": str(e)}, f, indent=4)