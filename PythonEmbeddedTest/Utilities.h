#pragma once

#include <iostream>
#include <Python.h>

namespace Utilities
{
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
}

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

    PyObject* LoadImageCV2(const char* ImagePath)
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

    PyObject* GetCornersFromGeneratedMask(PyObject* Image, PyObject* TestModel, int tolerance = 10, int perCorner = 21, Utilities::Coordinate<float> scale_factor = Utilities::Coordinate<float>(0.95f, 0.95f))
    {
        PyObject* getcornersfromgeneratedmask_func = PyObject_GetAttrString(mrcnn_module, "GetCornersFromGeneratedMask");

        if (getcornersfromgeneratedmask_func == nullptr)
        {
            PyErr_Print();
            return nullptr;
        }

        PyObject* scale_factor_ptr = Py_BuildValue("(ff)", scale_factor.x, scale_factor.y);

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
