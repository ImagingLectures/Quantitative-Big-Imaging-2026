import os
import zipfile
import urllib.request
import tarfile
import argparse
import shutil

# TODO: Update this link with the new Polybox zip link
POLYBOX_LINK = "https://www.polybox.ethz.ch/index.php/s/H78KSbbffLgXkTx/download"
DATA_ZIP = "polybox_data.zip"

# List of files that should be present after setup
# These are the files from polybox_manifest.txt
MANIFEST = [
    "Lectures/IsolineAverage/data/Accusand_50_70_lowsuction_0_6_hours_0.tif",
    "Lectures/Lecture-01/data/dc_0000.tif",
    "Lectures/Lecture-01/data/ob_0000.tif",
    "Lectures/Lecture-01/data/wood.npy",
    "Lectures/Lecture-01/data/wood_0000.tif",
    "Lectures/Lecture-01/figures/anders2023.jpeg",
    "Lectures/Lecture-01/figures/experimentdata.pdf",
    "Lectures/Lecture-01/figures/experimentdata.svg",
    "Lectures/Lecture-01/figures/pixelbucket.png",
    "Lectures/Lecture-01/figures/yougotdata.png",
    "Lectures/Lecture-01/movies/dk31_foam.mp4",
    "Lectures/Lecture-01/movies/lightfield.mp4",
    "Lectures/Lecture-02/figures/CIFAR10-examples.png",
    "Lectures/Lecture-02/figures/StoreFeatures.svg",
    "Lectures/Lecture-02/figures/celeb_dataset.png",
    "Lectures/Lecture-02/figures/segmentationCD.pdf",
    "Lectures/Lecture-02/figures/segmentationCD.png",
    "Lectures/Lecture-02/figures/segmentationCD.svg",
    "Lectures/Lecture-03/03-ImageEnhancement.pdf",
    "Lectures/Lecture-03/figures/BM3D_BlockMatching.png",
    "Lectures/Lecture-03/figures/BM3D_BlockMatching2.png",
    "Lectures/Lecture-03/figures/ComparingBM3D.png",
    "Lectures/Lecture-03/figures/DL-Denoising.png",
    "Lectures/Lecture-03/figures/filter_overview_gray.pdf",
    "Lectures/Lecture-03/figures/filter_overview_gray.png",
    "Lectures/Lecture-03/figures/filter_overview_gray.svg",
    "Lectures/Lecture-03/figures/hp_principle.png",
    "Lectures/Lecture-03/figures/imperfect_imaging_system.png",
    "Lectures/Lecture-03/figures/lp_principle.png",
    "Lectures/Lecture-03/figures/recon5s_0130.tif",
    "Lectures/Lecture-03/figures/verification_differences.png",
    "Lectures/Lecture-03/figures/verification_differences.svg",
    "Lectures/Lecture-03/movies/nldif_iter.mp4",
    "Lectures/Lecture-03/movies/nldif_iter.swf",
    "Lectures/Lecture-04/figures/duck/dc.tif",
    "Lectures/Lecture-04/figures/duck/duck90.tif",
    "Lectures/Lecture-04/figures/duck/neglognorm.tif",
    "Lectures/Lecture-04/figures/duck/normalized.tif",
    "Lectures/Lecture-04/figures/duck/ob.tif",
    "Lectures/Lecture-05/data/nct0440.tif",
    "Lectures/Lecture-05/data/nct0450.tif",
    "Lectures/Lecture-05/data/nct0460.tif",
    "Lectures/Lecture-05/data/nct0470.tif",
    "Lectures/Lecture-05/figures/guardededge_demo.png",
    "Lectures/Lecture-05/figures/qt_demo.png",
    "Lectures/Lecture-05/figures/segmentation_types.png",
    "Lectures/Lecture-06/data/mask.tif",
    "Lectures/Lecture-06/data/plateau_border.tif",
    "Lectures/Lecture-06/data/tofdata.npy",
    "Lectures/Lecture-06/figures/aggregate_isoline.png",
    "Lectures/Lecture-06/figures/example_poster.tif",
    "Lectures/Lecture-07/data/Cropped_prediction_8bit.npy",
    "Lectures/Lecture-07/data/Cropped_prediction_8bit.npy.zip",
    "Lectures/Lecture-07/data/NMC_90wt_2000bar_115.tif",
    "Lectures/Lecture-07/data/grains.npy",
    "Lectures/Lecture-07/data/grains.npy.zip",
    "Lectures/Lecture-07/data/ws_grains.npy",
    "Lectures/Lecture-07/data/ws_grains.npy.zip",
    "Lectures/Lecture-07/figures/networks.png",
    "Lectures/Lecture-08/figures/StoreFeatures.svg",
    "Lectures/Lecture-08/figures/edge_object.jpg",
    "Lectures/Lecture-09/figures/ct_scan.gif",
    "Lectures/Lecture-09/figures/timescales.pdf",
    "Lectures/Lecture-09/figures/timescales.svg",
    "Lectures/Lecture-09/figures/trackpy.png",
    "Lectures/Lecture-09/movies/WaterJet.m4v",
    "Lectures/Lecture-09/movies/dk31_foam.mp4",
    "Lectures/Lecture-09/movies/snow_frames.mp4",
    "Lectures/Lecture-09/movies/snow_tracks.mp4",
    "Lectures/Lecture-09/movies/snowfall.mp4",
    "Lectures/Lecture-10/figures/BiologyModalities.png",
    "Lectures/Lecture-10/figures/NX-OnTheFlySetup.svg",
    "Lectures/Lecture-10/figures/electronmicro.png",
    "Lectures/Lecture-10/figures/registration.png",
    "Lectures/Lecture-11/11-HCSBigData.pdf",
    "Lectures/Lecture-11/data/plateau_border.tif",
    "Lectures/Lecture-11/data/shakespeare.txt",
    "Lectures/Lecture-11/figures/An_illustration_of_the_dining_philosophers_problem.png",
    "Lectures/ext-figures/PaleontologyMovie.m4v",
    "Lectures/ext-figures/automaticthresh.tiff",
    "Lectures/ext-figures/cortex.psd",
    "Lectures/ext-figures/lecture03/lindiff.apng",
    "Projects/images/brainScan.gif",
    "Projects/student_presentations/QBI Final Presentation_Savina Kim.pdf",
    "Projects/student_presentations/Savorana_QBI_project.pdf",
    "Projects/student_presentations/qpi-cardiac-ethz.pdf",
    "common/data/full_img.csv",
    "common/data/wood.npy",
    "common/figures/PaleontologyMovie.m4v",
    "common/figures/cortex.psd",
    "common/figures/example_poster.tif",
    "common/movies/dk31-plat.avi",
    "common/movies/lightfield.mp4",
]

def download_data():
    if not os.path.exists(DATA_ZIP):
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        print(f"Downloading data from Polybox...")
        
        class DownloadProgressBar(tqdm if tqdm else object):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        if tqdm:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=DATA_ZIP) as t:
                urllib.request.urlretrieve(POLYBOX_LINK, filename=DATA_ZIP, reporthook=t.update_to)
        else:
            urllib.request.urlretrieve(POLYBOX_LINK, DATA_ZIP)
            
        print("Download complete.")
    else:
        print("Data zip already exists, skipping download.")

def extract_and_move():
    print("Extracting Polybox data...")
    # First extraction
    with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Check if we found another polybox_data.zip in a subfolder (Polybox folder download behavior)
    for root, dirs, files in os.walk("."):
        if DATA_ZIP in files and root != ".":
            nested_zip = os.path.join(root, DATA_ZIP)
            print(f"Found nested zip: {nested_zip}. Extracting...")
            with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
                zip_ref.extractall(".")
            # Optional: remove the nested zip and the folder it was in
            break

def main():
    parser = argparse.ArgumentParser(description="Setup data for QBI exercises")
    parser.add_argument("--check", action="store_true", help="Check if data is missing")
    parser.add_argument("--clean", action="store_true", help="Delete all downloaded and uncompressed data")
    args = parser.parse_args()

    if args.clean:
        print("Cleaning up data...")
        if os.path.exists(DATA_ZIP):
            print(f"Removing {DATA_ZIP}")
            os.remove(DATA_ZIP)
        
        # Also remove extracted subfolders if they exist
        if os.path.exists("lectures_data"):
             shutil.rmtree("lectures_data")

        for path in MANIFEST:
            if os.path.exists(path):
                print(f"Removing {path}")
                os.remove(path)

        print("Cleanup complete.")
        return

    if args.check:
        missing = False
        for path in MANIFEST:
            if not os.path.exists(path):
                print(f"Missing: {path}")
                missing = True
        if not missing:
            print("All big files are present.")
        return

    download_data()
    extract_and_move()
    print("Setup complete.")

if __name__ == "__main__":
    main()

