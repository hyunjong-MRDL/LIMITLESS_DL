import slicer
import csv

def calculate_and_save_dice(segmentNode1, segmentNode2, outputCsvPath):
    """
    Automatically matches ROIs with identical names in two segmentation nodes,
    calculates Dice coefficients, and saves the results to a CSV file.

    Parameters:
        segmentNode1 (vtkMRMLSegmentationNode): First segmentation node.
        segmentNode2 (vtkMRMLSegmentationNode): Second segmentation node.
        outputCsvPath (str): Path to save the Dice coefficients CSV file.
    """
    # Get segment names from both nodes
    segmentIDs1 = [segmentNode1.GetSegmentation().GetNthSegmentID(i)
                   for i in range(segmentNode1.GetSegmentation().GetNumberOfSegments())]
    segmentIDs2 = [segmentNode2.GetSegmentation().GetNthSegmentID(i)
                   for i in range(segmentNode2.GetSegmentation().GetNumberOfSegments())]

    # Match segments by names
    matchingSegments = []
    for segmentID1 in segmentIDs1:
        name1 = segmentNode1.GetSegmentation().GetSegment(segmentID1).GetName()
        for segmentID2 in segmentIDs2:
            name2 = segmentNode2.GetSegmentation().GetSegment(segmentID2).GetName()
            if name1 == name2:  # Match by identical names
                matchingSegments.append((segmentID1, segmentID2))
                break

    if not matchingSegments:
        slicer.util.errorDisplay("No matching ROIs with identical names found.")
        return

    # Perform Dice coefficient computation
    segmentComparisonLogic = slicer.modules.segmentcomparison.logic()
    results = []

    for segmentID1, segmentID2 in matchingSegments:
        name = segmentNode1.GetSegmentation().GetSegment(segmentID1).GetName()
        print(f"Calculating Dice for segment: {name}")

        # Run the Segment Comparison logic
        diceResult = segmentComparisonLogic.ComputeDice(
            segmentNode1.GetID(), segmentID1,
            segmentNode2.GetID(), segmentID2
        )

        results.append([name, diceResult])

    # Save results to a CSV file
    with open(outputCsvPath, mode='w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["ROI Name", "Dice Coefficient"])
        writer.writerows(results)

    slicer.util.infoDisplay(f"Dice coefficients saved to {outputCsvPath}")

# Example usage in 3D Slicer
segmentNode1 = slicer.mrmlScene.GetFirstNodeByName("SpinalCord")  # Replace with your node name
segmentNode2 = slicer.mrmlScene.GetFirstNodeByName("SpinalCord")  # Replace with your node name
outputCsvPath = "E:/CT_Recon/dice1.csv"  # Update to your preferred save location

calculate_and_save_dice(segmentNode1, segmentNode2, outputCsvPath)