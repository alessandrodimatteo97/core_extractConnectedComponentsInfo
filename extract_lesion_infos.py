import SimpleITK as sitk
import csv
import os
import numpy as np

def get_components(image, min_size, full_connectivity=True):
    """Extract connected components, compute lesion volumes and coordinates."""
    ccifilter = sitk.ConnectedComponentImageFilter()
    ccifilter.SetFullyConnected(full_connectivity)
    labeled = ccifilter.Execute(image)

    rcif = sitk.RelabelComponentImageFilter()
    rcif.SetMinimumObjectSize(min_size) 
    labeled = rcif.Execute(labeled)

    labeled_data = sitk.GetArrayFromImage(labeled)
    ncomponents = rcif.GetNumberOfObjects()

    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(labeled, labeled)

    spacing = image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    lesion_info = []

    for label in range(1, ncomponents + 1):
        voxel_count = lsif.GetCount(label)
        volume_mm3 = voxel_count * voxel_volume

        coords = np.argwhere(labeled_data == label)  # (z, y, x)
        centroid_voxel = coords.mean(axis=0)  # (z, y, x) in voxel space

        # Convert to physical point (x, y, z in mm)
        centroid_index = tuple(int(round(x)) for x in centroid_voxel[::-1])  # (x, y, z) index
        centroid_mm = image.TransformIndexToPhysicalPoint(centroid_index)

        lesion_info.append((label, voxel_count, volume_mm3,
                            tuple(centroid_voxel[::-1]),  # (x, y, z) voxel
                            centroid_mm))                # (x_mm, y_mm, z_mm)

    return labeled_data, ncomponents, lesion_info


def save_to_csv(filename, lesion_volumes):
    """Saves lesion volume and coordinate data to a CSV file."""
    lesion_volumes.sort(key=lambda x: x[0])  # Sort by subject

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Subject", "Modality", "Lesion ID", "Voxel Count", "Volume (mm³)",
            "Centroid Voxel (x,y,z)", "Centroid Physical (x_mm, y_mm, z_mm)"
        ])
        for row in lesion_volumes:
            subject, modality, label, vcount, vol, voxel_centroid, mm_centroid = row
            writer.writerow([
                subject, modality, label, vcount, vol,
                f"{voxel_centroid[0]:.2f},{voxel_centroid[1]:.2f},{voxel_centroid[2]:.2f}",
                f"{mm_centroid[0]:.2f},{mm_centroid[1]:.2f},{mm_centroid[2]:.2f}"
            ])

    print(f"Lesion data saved to {filename}")


# Main Execution
if __name__ == "__main__":
    for min_size in [10]:
        print(min_size)
        root_directory = "./derivatives/"
        output_csv = f"segmentation_results_derivatives_ATLAS_noprep_{min_size}_mm_with_centroid.csv"
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


            if gt_file_T1:
                image_T1 = sitk.ReadImage(gt_file_T1, sitk.sitkFloat64)  # Load the image
                image_T1 = sitk.BinaryThreshold(image_T1,
                                        lowerThreshold=0.00001, 
                                        upperThreshold=1e6, 
                                        insideValue=1, 
                                        outsideValue=0)
                labeled_data, ncomponents, lesion_volumes_T1 = get_components(image_T1, min_size, full_connectivity=True)

                # Append data correctly
                if len(lesion_volumes_T1) == 0: 
                    lesion_volumes_.append((subject, "T1", 0))
                else:
                    for lesion in lesion_volumes_T1:
                        label, vcount, vol, voxel_centroid, mm_centroid = lesion
                        lesion_volumes_.append((subject, "T1", label, vcount, vol, voxel_centroid, mm_centroid))




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
