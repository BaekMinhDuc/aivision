// AIVision 🚀 AGPL-3.0 License - https://aivision.com/license

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference.hpp"
#include "aivi_cli.hpp"
#include "aivi_render.hpp"
#include "aivi_show.hpp"

int main(int argc, char** argv) {
    aivi::Config config;
    config.url = aivi::ArgValue(argc, argv, "--url", "localhost:8001");
    config.model = aivi::ArgValue(argc, argv, "--model", "aivi26n");
    config.model_version = aivi::ArgValue(argc, argv, "--version", "");
    config.conf = std::stof(aivi::ArgValue(argc, argv, "--conf", "0.25"));
    config.iou = std::stof(aivi::ArgValue(argc, argv, "--iou", "0.45"));
    config.imgsz = std::stoi(aivi::ArgValue(argc, argv, "--imgsz", "640"));
    config.task = aivi::TaskFromString(aivi::ArgValue(argc, argv, "--task", "unknown"));
    const std::string source = aivi::ArgValue(argc, argv, "--source", "bus.jpg");
    const std::string output = aivi::ArgValue(argc, argv, "--out", "result.jpg");
    const bool show = aivi::ShowRequested(argc, argv);

    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        std::cerr << "ERROR: could not read image '" << source << "'" << std::endl;
        return 1;
    }

    try {
        aivi::Predictor predictor(config);
        const std::vector<std::string>& names = predictor.names();

        cv::Mat semantic;
        std::vector<aivi::Result> results = predictor.predict(image, semantic);
        std::cout << "Model: " << config.model << " @ " << config.url
                  << " | task: " << aivi::TaskName(predictor.task()) << " | classes: " << names.size() << std::endl;

        cv::Mat canvas = image.clone();
        aivi::RenderAndPrint(canvas, predictor.task(), results, names, semantic);

        cv::imwrite(output, canvas);
        std::cout << "Result image written to " << output << std::endl;
        aivi::Show("AIVI", canvas, show);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
