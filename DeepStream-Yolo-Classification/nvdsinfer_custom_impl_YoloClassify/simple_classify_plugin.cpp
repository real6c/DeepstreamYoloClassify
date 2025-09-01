/*
 * Simple YOLO Classification Plugin for DeepStream
 */

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstring>

// Plugin initialization
__attribute__((constructor))
void plugin_init() {
    std::cout << "INFO: YOLO Classification Plugin loaded successfully" << std::endl;
}

__attribute__((destructor))
void plugin_cleanup() {
    // Plugin unloaded
}

// Structure to hold classification results
struct ClassificationResult {
    int classId;
    float confidence;
    std::string className;
};

// Global variable to store class labels - will be loaded from file
static std::vector<std::string> g_classLabels;

// Function to load labels from file
static bool loadLabelsFromFile(const std::string& filename) {
    // Try multiple possible paths for the labels file
    std::vector<std::string> possiblePaths = {
        filename,                                    // Current directory
        "../" + filename,                           // Parent directory
        "../../" + filename,                        // Grandparent directory
        "/home/jetson/DeepStreamYoloCls/" + filename, // Absolute path to workspace
        "/home/jetson/DeepStreamYoloCls/DeepStream-Yolo-Classification/" + filename, // Absolute path to project
        "./DeepStream-Yolo-Classification/" + filename, // Relative to current
        "./nvdsinfer_custom_impl_YoloClassify/" + filename // Plugin directory
    };
    
    for (const auto& path : possiblePaths) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::cout << "INFO: Successfully opened labels file: " << path << std::endl;
            
            g_classLabels.clear();
            std::string line;
            while (std::getline(file, line)) {
                // Remove whitespace and empty lines
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                
                if (!line.empty()) {
                    g_classLabels.push_back(line);
                }
            }
            
            file.close();
            
            if (!g_classLabels.empty()) {
                std::cout << "INFO: Loaded " << g_classLabels.size() << " labels from: " << path << std::endl;
                return true;
            } else {
                std::cerr << "WARNING: File opened but no labels found in: " << path << std::endl;
            }
        } else {
            std::cout << "DEBUG: Could not open labels file: " << path << std::endl;
        }
    }
    
    std::cerr << "ERROR: Failed to load labels from any of the attempted paths" << std::endl;
    return false;
}

// Function to get top-k classifications
static std::vector<ClassificationResult> getTopKClassifications(
    const float* output, 
    int numClasses, 
    int topK = 5,
    float threshold = 0.1f) {
    
    std::vector<std::pair<float, int>> scores;
    scores.reserve(numClasses);
    
    // Create pairs of (confidence, class_id)
    for (int i = 0; i < numClasses; ++i) {
        scores.push_back(std::make_pair(output[i], i));
    }
    
    // Sort by confidence (descending)
    std::sort(scores.begin(), scores.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    std::vector<ClassificationResult> results;
    for (int i = 0; i < std::min(topK, numClasses); ++i) {
        if (scores[i].first >= threshold) {
            ClassificationResult result;
            result.classId = scores[i].second;
            result.confidence = scores[i].first;
            result.className = (scores[i].second < g_classLabels.size()) ? 
                              g_classLabels[scores[i].second] : 
                              "Unknown_" + std::to_string(scores[i].second);
            results.push_back(result);
        }
    }
    
    return results;
}

// Function to get and print the current working directory
static void printWorkingDirectory() {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        std::cerr << "ERROR: Could not get current working directory." << std::endl;
    }
}

// Main classification parsing function
extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                          NvDsInferNetworkInfo const& networkInfo,
                          NvDsInferParseDetectionParams const& detectionParams,
                          std::vector<NvDsInferParseObjectInfo>& objectList) {
    
    // Load labels if not already loaded (first time this function is called)
    if (g_classLabels.empty()) {
        std::cout << "INFO: Attempting to load labels from file..." << std::endl;
        printWorkingDirectory();
        
        if (!loadLabelsFromFile("labels.txt")) {
            std::cerr << "ERROR: Failed to load labels from file, using fallback labels" << std::endl;
            std::cout << "INFO: Loading " << 38 << " fallback labels..." << std::endl;
            // Fallback to hardcoded labels if file loading fails, this shouldn't run ever, but if you need to you can hardcode your labels here
            g_classLabels = {
                "cardboard_box", "glass_bottle", "glass_container", "glass_decor", "glass_dishes",
                "glass_drinking", "glass_jar", "glass_lab", "glass_other", "metal_can",
                "metal_cap", "metal_foil", "metal_other", "metal_small_items", "other_paper",
                "paper_bag", "paper_bowl", "paper_carton", "paper_cup", "paper_magazine",
                "paper_mail", "paper_napkin", "paper_plate", "paper_receipt", "paper_sheet",
                "paper_straw", "paper_wrapper", "plastic_bag", "plastic_bottle", "plastic_container",
                "plastic_cup", "plastic_jug", "plastic_lid", "plastic_other", "plastic_packaging",
                "plastic_soap_bottle", "plastic_straw", "plastic_tube", "plastic_utensils"
            };
            std::cout << "INFO: Fallback labels loaded successfully" << std::endl;
        } else {
            std::cout << "INFO: Labels loaded successfully from file" << std::endl;
        }
    }
    
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: Could not find output layer in classification parsing" << std::endl;
        return false;
    }
    
    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    const float* outputBuffer = static_cast<const float*>(output.buffer);
    
    if (!outputBuffer) {
        std::cerr << "ERROR: Output buffer is null!" << std::endl;
        return false;
    }
    
    // Get number of classes from output dimensions
    int numClasses = output.inferDims.d[0];
    
    // Get top classifications with reasonable threshold
    std::vector<ClassificationResult> classifications = getTopKClassifications(
        outputBuffer, numClasses, 5, 0.01f);
    
    // Convert to DeepStream object format
    // For classification, we create a full-frame object with the top classification
    if (!classifications.empty()) {
        NvDsInferParseObjectInfo obj;
        
        // Set bounding box to cover the entire frame
        obj.left = 0.0f;
        obj.top = 0.0f;
        obj.width = static_cast<float>(networkInfo.width);
        obj.height = static_cast<float>(networkInfo.height);
        
        // Set classification info
        obj.classId = classifications[0].classId;
        obj.detectionConfidence = classifications[0].confidence;
        
        objectList.push_back(obj);
        
        // Print top classification result
        std::cout << "Classification: " << classifications[0].className 
                  << " (Confidence: " << std::fixed << std::setprecision(3) << classifications[0].confidence << ")" << std::endl;
    }
    
    return true;
}

// Register the function
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
