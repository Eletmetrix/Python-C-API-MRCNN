
#include <iostream>
#include <Python.h>
#include <vector>
#include <sstream>

/*
 * Coordinate structure
 */
struct Coordinate
{
    uint16_t x;
    uint16_t y;

    Coordinate()
        : x(0), y(0)
    { }

    Coordinate(uint16_t X, uint16_t Y)
        : x(X), y(Y)
    { }

    std::string toString() const
    {
        return "(X: " + std::to_string(x) + ", Y: " + std::to_string(y) + ")";
    }
};

struct Rectangle
{
    Coordinate LeftUp;
    Coordinate RightDown;

    Rectangle()
        : LeftUp(), RightDown()
    {}

    Rectangle(Coordinate leftUp, Coordinate rightDown)
        : LeftUp(leftUp), RightDown(rightDown)
    {}

    std::string toString() const
    {
        return "(Left Up Corner: " + LeftUp.toString() + ", Right Down Corner: " + RightDown.toString() + ")";
    }
};

std::ostream& operator<< (std::ostream& os, const Coordinate& coord)
{
    os << "(" << coord.x << ", " << coord.y << ")";
    return os;
}

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

    PyObject* GetCornersFromGeneratedMask(PyObject* Image, PyObject* TestModel, int tolerance = 10, int perCorner = 21, Coordinate scale_factor = Coordinate(0.95f, 0.95f))
    {
        PyObject* getcornersfromgeneratedmask_func = PyObject_GetAttrString(mrcnn_module, "GetCornersFromGeneratedMask");

        if (getcornersfromgeneratedmask_func == nullptr)
        {
            PyErr_Print();
            return nullptr;
        }

        PyObject* Coords = PyObject_CallObject(getcornersfromgeneratedmask_func, PyTuple_Pack(5, 
            Image, 
            TestModel, 
            PyLong_FromLong(tolerance), 
            PyLong_FromLong(perCorner), 
            PyTuple_Pack(2, 
                PyLong_FromLong(scale_factor.x), 
                PyLong_FromLong(scale_factor.y))));

        if (Coords == nullptr)
        {
            PyErr_Print();
            return nullptr;
        }

        Py_DECREF(getcornersfromgeneratedmask_func);
        return Coords;
    }

protected:
    PyObject* mrcnn_module = nullptr;
};

static Rectangle rect = Rectangle();

int main()
{
    Py_Initialize();

    // Load libraries.
    MaskRCNN* mrcnn = new MaskRCNN;

    if (!mrcnn->IsValid())
    {
        // Modules didn't loaded! We can't continue with that issue.
        return EXIT_FAILURE;
    }

    PyObject* Image = mrcnn->LoadImage("Images\\image.jpg");
    PyObject* TestModel = mrcnn->LoadReadyWeights("Data\\mask_rcnn_object_0005.h5");
    PyObject* Coords = mrcnn->GetCornersFromGeneratedMask(Image, TestModel);

    Py_DECREF(Image);
    Py_DECREF(TestModel);

    /*PyObject* pStr = PyObject_Repr(Coords);
    if (Coords)
    {
        const char* str = PyUnicode_AsUTF8(pStr);
        std::cout << "Val: " << str << std::endl;
    }*/

    rect = Rectangle(
        Coordinate(PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 0), 0)), PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 1), 0))),
        Coordinate(PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 0), 1)), PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(Coords, 1), 1)))
    );

    std::cout << rect.toString();

    Py_DECREF(Coords);

    Py_Finalize();

    return EXIT_SUCCESS;
}
