
#include <iostream>
#include <Python.h>
#include <vector>
#include <sstream>

template<typename T>
struct Coordinate
{
    T x;
    T y;

    Coordinate()
        : x(0), y(0)
    { }

    Coordinate(T X, T Y)
        : x(X), y(Y)
    { }

    std::string toString() const
    {
        return "(X: " + std::to_string(x) + ", Y: " + std::to_string(y) + ")";
    }
};

struct Rectangle
{
    Coordinate<uint16_t> LeftUp;
    Coordinate<uint16_t> RightDown;

    Rectangle()
        : LeftUp(), RightDown()
    {}

    Rectangle(Coordinate<uint16_t> leftUp, Coordinate<uint16_t> rightDown)
        : LeftUp(leftUp), RightDown(rightDown)
    {}

    std::string toString() const
    {
        return "(Left Up Corner: " + LeftUp.toString() + ", Right Down Corner: " + RightDown.toString() + ")";
    }
};

/*
 * The Mask R-CNN library. 
 */
class MaskRCNN
{
public:
    /*
     * The constructor.
     */
    MaskRCNN()
    {
        // sys.path.
        PyObject* sys_path = PySys_GetObject("path");
        if (sys_path == NULL || !PyList_Check(sys_path)) {
            std::cout << "sys.path is not a list." << std::endl;
            return;
        }

        // add mask_rcnn path to sys.path.
        PyObject* new_dir = Py_BuildValue("s", "Python\\mask_rcnn");
        int result = PyList_Append(sys_path, new_dir);
        if (result == -1) {
            std::cout << "cannot append to sys.path." << std::endl;
            return;
        }

        // Initialize module variables.
        mrcnn_module = PyImport_ImportModule("m_rcnn");

        if (!mrcnn_module) {
            PyErr_Print();
            std::cout << "modules are not found!" << std::endl;
            mrcnn_module = nullptr;
        }
    }

    /*
     * The destructor.
     */
    ~MaskRCNN()
    {
        Py_XDECREF(mrcnn_module);

        mrcnn_module = nullptr;
    }

    /*
     * This is for checking our module was loaded successfully.
     */
    bool IsValid() const
    {
        return mrcnn_module != nullptr;
    }

    PyObject* LoadImage(const char* ImagePath)
    {
        PyObject* loadimage_func = PyObject_GetAttrString(mrcnn_module, "LoadImage");
        PyObject* Image = PyObject_CallObject(loadimage_func, PyTuple_Pack(1, PyUnicode_FromString(ImagePath)));
        Py_DECREF(loadimage_func);
        return Image;
    }

    PyObject* LoadReadyWeights(const char* WeightPath)
    {
        PyObject* loadreadyweights_func = PyObject_GetAttrString(mrcnn_module, "LoadReadyWeights");
        PyObject* TestModel = PyObject_CallObject(loadreadyweights_func, PyTuple_Pack(1, PyUnicode_FromString(WeightPath)));
        Py_DECREF(loadreadyweights_func);
        return TestModel;
    }

    PyObject* GetCornersFromGeneratedMask(PyObject* Image, PyObject* TestModel, int tolerance = 10, int perCorner = 21, Coordinate<float> scale_factor = Coordinate<float>(0.95f, 0.95f))
    {
        PyObject* getcornersfromgeneratedmask_func = PyObject_GetAttrString(mrcnn_module, "GetCornersFromGeneratedMask");

        if (getcornersfromgeneratedmask_func == nullptr)
        {
            PyErr_Print();
            return nullptr;
        }

        PyObject* scale_factor_ptr = Py_BuildValue("(ff)", scale_factor.x, scale_factor.y);

        std::cout << "CPP: " << scale_factor.toString() << ", Python: " << PyUnicode_AsUTF8(PyObject_Repr(scale_factor_ptr));

        PyObject* Coords = PyObject_CallObject(getcornersfromgeneratedmask_func, PyTuple_Pack(5, 
            Image, 
            TestModel, 
            PyLong_FromLong(tolerance), 
            PyLong_FromLong(perCorner), 
            scale_factor_ptr));

        if (Coords == nullptr)
        {
            PyErr_Print();
            return nullptr;
        }

        Py_DECREF(getcornersfromgeneratedmask_func);
        return Coords;
    }

    void SaveImage(PyObject* Image, PyObject* Rect, const char* Path)
    {
        PyObject* saveimage_func = PyObject_GetAttrString(mrcnn_module, "SaveImage");

        if (saveimage_func == nullptr)
        {
            PyErr_Print();
            return;
        }

        PyObject_CallObject(saveimage_func, PyTuple_Pack(3, Image, Rect, PyUnicode_FromString(Path)));
    }

protected:
    PyObject* mrcnn_module = nullptr;
};

static Rectangle rect = Rectangle();

int main()
{
    std::cout << "System Initializing..." << std::endl;

    Py_Initialize();

    // Load libraries.
    MaskRCNN* mrcnn = new MaskRCNN;

    if (!mrcnn->IsValid())
    {
        // Modules didn't loaded! We can't continue with that issue.
        return EXIT_FAILURE;
    }

    PyObject* TestModel = mrcnn->LoadReadyWeights("Data\\mask_rcnn_object_0005.h5");

comeback:

    std::string file_path = "Images\\image.jpg";
    while (true)
    {
        std::cout << "Please enter the file name [Images\\image.jpg]: ";
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty())
        {
            file_path = input;
        }

        FILE* file;
        if (fopen_s(&file, file_path.c_str(), "r") == 0)
        {
            fclose(file);
            break;
        }
        else
        {
            std::cout << "File doesn't exists. Make sure your file is valid." << std::endl;
            file_path = "Images\\image.jpg";
        }
    }

    PyObject* Image = mrcnn->LoadImage(file_path.c_str());
    PyObject* Coords = mrcnn->GetCornersFromGeneratedMask(Image, TestModel);

    Py_DECREF(Image);
    Py_DECREF(TestModel);

    rect = Rectangle(
        Coordinate<uint16_t>(PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 0), 0)), PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 1), 0))),
        Coordinate<uint16_t>(PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 0), 1)), PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 1), 1)))
    );

    std::cout << rect.toString() << std::endl;

    std::cout << "Would you like to save the image? (y/n): ";
    std::string answer;
    std::getline(std::cin, answer);
    if (!answer.empty() && (answer == "y" || answer == "Y"))
    {
        std::string output = file_path;

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

    Py_DECREF(Coords);

    std::cout << "Would you like to enter a new file name? (y/n): ";
    std::getline(std::cin, answer);
    if (!answer.empty() && (answer == "y" || answer == "Y"))
    {
        goto comeback;
    }

    Py_Finalize();

    return EXIT_SUCCESS;
}
