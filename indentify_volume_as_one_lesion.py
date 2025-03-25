import SimpleITK as sitk
import csv
import os
import nibabel as nib
import numpy as np

def get_just_one_component(volume):
    return np.count_nonzero(volume)


def get_components(image, min_size, full_connectivity=True):
    """Extract connected components and compute lesion volumes."""
    ccifilter = sitk.ConnectedComponentImageFilter()
    ccifilter.SetFullyConnected(full_connectivity)  # 26-connectivity if True, else 6-connectivity
    labeled = ccifilter.Execute(image)

    rcif = sitk.RelabelComponentImageFilter()
    labeled = rcif.Execute(labeled)

    labeled_data = sitk.GetArrayFromImage(labeled)
    ncomponents = rcif.GetNumberOfObjects()

    if ncomponents != ccifilter.GetObjectCount():
        print('Number of components mismatch!')
        print(ccifilter.GetObjectCount())

    # Compute lesion volumes
    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(labeled, labeled)

    spacing = image.GetSpacing()  # Get voxel spacing (dx, dy, dz)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Volume of a single voxel in mm³

    lesion_volumes = []
    for label in range(1, ncomponents + 1):  # Labels start from 1
        voxel_count = lsif.GetCount(label)  # Number of voxels in this lesion
        volume_mm3 = voxel_count * voxel_volume  # Convert to mm³
        lesion_volumes.append((label, voxel_count, volume_mm3))

    return labeled_data, ncomponents, lesion_volumes

def save_to_csv(filename, lesion_volumes):
    """Saves lesion volume data to a CSV file, sorted by subject."""
    
    # Ordinamento della lista in base al soggetto
    lesion_volumes.sort(key=lambda x: x[0])  # Ordina per "Subject" (primo elemento della tupla)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Subject", "Modality", "Voxel Count"])  # Header row
        writer.writerows(lesion_volumes)
    
    print(f"Lesion volume data saved to {filename}")

# Main Execution
if __name__ == "__main__":
    root_directory = "./derivatives/"
    output_csv = "segmentation_results_derivatives_ATLAS_just_one_lesion_final.csv"
    lesion_volumes_ = []

    for subject in os.listdir(root_directory):
        subject_path = os.path.join(root_directory, subject)
        seg_folder = os.path.join(subject_path, "seg")

        if not os.path.isdir(seg_folder):
            continue

        gt_file_FLAIR = None
        gt_file_T1 = None
        for file in os.listdir(seg_folder):
            if file.endswith("_seg.nii.gz"): # sub-01001/seg/sub-01001_T1w_reg_clos.nii.gz
                gt_file_T1 = os.path.join(seg_folder, file)
            """if file.endswith("FLAIR_reg_fh.nii.gz"):
                gt_file_FLAIR = os.path.join(seg_folder, file)"""

        min_size = 0

        if gt_file_T1:
            volume = nib.load(gt_file_T1).get_fdata()
            #labeled_data, ncomponents, lesion_volumes_T1 = get_components(volume)
            lesion = get_just_one_component(volume)
            lesion_volumes_.append((subject, "T1", str(lesion)))

        # Process FLAIR lesions
        """if gt_file_FLAIR:
            image_FLAIR = sitk.ReadImage(gt_file_FLAIR, sitk.sitkFloat64)  # Load the image
            image_FLAIR  = sitk.BinaryThreshold(image_FLAIR,
                                    lowerThreshold=0.500000001, 
                                    upperThreshold=1e6, 
                                    insideValue=1, 
                                    outsideValue=0)
            
            labeled_data, ncomponents, lesion_volumes_FLAIR = get_components(image_FLAIR, min_size)

            # Append data correctly
            for lesion in lesion_volumes_FLAIR:
                lesion_volumes_.append((subject, "FLAIR") + lesion)"""

    # Save results to CSV with sorted order
    save_to_csv(output_csv, lesion_volumes_)

    print(f"Processed all subjects. Results saved to {output_csv}")
