// AIVision 🚀 AGPL-3.0 License - https://aivision.com/license

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference.h"
#include "aivi_cli.hpp"
#include "aivi_render.hpp"
#include "aivi_show.hpp"

int main(int argc, char** argv) {
    aivi::Config config;
    config.model_path = aivi::ArgValue(argc, argv, "--model", "aivi26n.onnx");
    config.conf = std::stof(aivi::ArgValue(argc, argv, "--conf", "0.25"));
    config.iou = std::stof(aivi::ArgValue(argc, argv, "--iou", "0.45"));
    config.device = aivi::ArgValue(argc, argv, "--device", "AUTO");
    const std::string source = aivi::ArgValue(argc, argv, "--source", "bus.jpg");
    const std::string output = aivi::ArgValue(argc, argv, "--out", "result.jpg");
    const bool show = aivi::ShowRequested(argc, argv);

    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        std::cerr << "ERROR: could not read image '" << source << "'" << std::endl;
        return 1;
    }

    aivi::Predictor predictor(config);
    const std::vector<std::string>& names = predictor.names();
    std::cout << "Model: " << config.model_path << " | task: " << aivi::TaskName(predictor.task())
              << " | classes: " << names.size() << std::endl;

    cv::Mat semantic;
    std::vector<aivi::Result> results = predictor.predict(image, semantic);

    cv::Mat canvas = image.clone();
    aivi::RenderAndPrint(canvas, predictor.task(), results, names, semantic);

    cv::imwrite(output, canvas);
    std::cout << "Result image written to " << output << std::endl;
    aivi::Show("AIVI", canvas, show);
    return 0;
}
