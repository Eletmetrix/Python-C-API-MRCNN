
#include <sstream>

#include "Utilities.h"

#include <boost/program_options.hpp>
#include <Windows.h>

namespace po = boost::program_options;

static Utilities::Rectangle rect = Utilities::Rectangle();

int wmain(int argc, wchar_t* argv[])
{
    std::vector<std::string> args_utf8;
    for (int i = 0; i < argc; i++) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, argv[i], wcslen(argv[i]), NULL, 0, NULL, NULL);
        std::string arg_utf8(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, argv[i], wcslen(argv[i]), &arg_utf8[0], size_needed, NULL, NULL);
        args_utf8.push_back(arg_utf8);
    }

    po::options_description desc("TeamCyberless's Mask-RCNN Library Test.\nUsage:\n  mrcnn [OPTION]...\n\nOptions");
    desc.add_options()
        ("version,V", "Display the version of this program and exit.")
        ("help,h", "Print this help.")
        ("quiet,q", "Do not output any message")
        ("no-interaction,n", "Do not ask any interactive question")
        ("save-file,s", "Save output image. Can be used with --no-interaction only")
        ("input,i", po::value<std::string>(), "Get single image file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl << "Examples:" << std::endl;
        std::cout << "  mrcnn -n -i Images/image.jpg" << std::endl;
        return 0;
    }

    if (vm.count("version")) {
        std::cout << "Version: 1.0" << std::endl;
        return 0;
    }

    std::cout << "System Initializing..." << std::endl;

    Py_Initialize();

    std::string input_file;
    if (vm.count("input")) {
        input_file = vm["input"].as<std::string>();

        FILE* file;
        if (fopen_s(&file, input_file.c_str(), "r") == 0)
        {
            fclose(file);
        }
        else
        {
            std::cout << "File doesn't exists. Make sure your file is valid." << std::endl;
            return 1;
        }
    }

    if (vm.count("quiet")) {
        // @TODO: Disable all python prints.
        std::cout << "Quiet mode didn\'t implemented yet. Please use without \"--quiet\" argument" << std::endl;
        return 1;
    }

    bool no_interaction = false;
    if (vm.count("no-interaction")) {
        no_interaction = true;
        if (vm.count("save-file")) {
            std::cout << "Save option didn\'t implemented yet. Please use without \"--save-file\" argument" << std::endl;
        }
    } 

    // Load libraries.
    MaskRCNN* mrcnn = new MaskRCNN;

    if (!mrcnn->IsValid())
    {
        // Modules didn't loaded! We can't continue with that issue.
        return EXIT_FAILURE;
    }

    PyObject* TestModel = mrcnn->LoadReadyWeights("Data\\mask_rcnn_object_0005.h5");

comeback:

    if (input_file.empty())
    {
        input_file = "Images\\image.jpg";
    }

    while (!no_interaction)
    {
        std::cout << "Please enter the file name [Images\\image.jpg]: ";
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty())
        {
            input_file = input;
        }

        FILE* file;
        if (fopen_s(&file, input_file.c_str(), "r") == 0)
        {
            fclose(file);
            break;
        }
        else
        {
            std::cout << "File doesn't exists. Make sure your file is valid." << std::endl;
            input_file = "Images\\image.jpg";
        }
    }

    PyObject* Image = mrcnn->LoadImageCV2(input_file.c_str());
    PyObject* Coords = mrcnn->GetCornersFromGeneratedMask(Image, TestModel);

    Py_DECREF(Image);
    Py_DECREF(TestModel);

    rect = Utilities::Rectangle(
        Utilities::Coordinate<uint16_t>(PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 0), 0)), PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 1), 0))),
        Utilities::Coordinate<uint16_t>(PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 0), 1)), PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 1), 1)))
    );

    std::cout << rect.toString() << std::endl;

    if (!no_interaction)
    {
        std::cout << "Would you like to save the image? (y/n): ";
        std::string answer;
        std::getline(std::cin, answer);
        if (!answer.empty() && (answer == "y" || answer == "Y"))
        {
            std::string output = input_file;

            size_t pos = output.find_last_of("\\/");
            std::string file_name = (pos == std::string::npos) ? output : output.substr(pos + 1);
            pos = file_name.find_last_of(".");
            std::string file_ext = (pos == std::string::npos) ? "" : file_name.substr(pos);

            std::string new_file_name = "output" + file_ext;
            output.replace(output.end() - file_name.length(), output.end(), new_file_name);

            mrcnn->SaveImage(Image, Coords, output.c_str());

            PyErr_Print();
        }

        answer.clear();

        std::cout << "Would you like to enter a new file name? (y/n): ";
        std::getline(std::cin, answer);
        if (!answer.empty() && (answer == "y" || answer == "Y"))
        {
            goto comeback;
        }
    }

    if (!no_interaction)
    {
        std::string output = input_file;

        size_t pos = output.find_last_of("\\/");
        std::string file_name = (pos == std::string::npos) ? output : output.substr(pos + 1);
        pos = file_name.find_last_of(".");
        std::string file_ext = (pos == std::string::npos) ? "" : file_name.substr(pos);

        std::string new_file_name = "output" + file_ext;
        output.replace(output.end() - file_name.length(), output.end(), new_file_name);

        mrcnn->SaveImage(Image, Coords, output.c_str());

        PyErr_Print();
    }

    Py_DECREF(Coords);

    Py_Finalize();

    return EXIT_SUCCESS;
}
