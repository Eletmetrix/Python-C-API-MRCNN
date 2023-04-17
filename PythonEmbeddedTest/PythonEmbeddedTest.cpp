
#include <iostream>
#include <Python.h>

// Our macros for exit statuses.
#define EXIT_SUCCESS 0;
#define EXIT_FAILURE 1;

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
        visualize_module = PyImport_ImportModule("visualize");

        if (!mrcnn_module || !visualize_module) {
            std::cout << "modules are not found!" << std::endl;
            mrcnn_module = nullptr;
            visualize_module = nullptr;
            return;
        }

        // Initialize functions.
        load_inference_model_func = PyObject_GetAttrString(mrcnn_module, "load_inference_model");
        if (load_inference_model_func == nullptr || !PyCallable_Check(load_inference_model_func)) {
            std::cout << "load_inference_model() not found or not callable." << std::endl;
            return;
        }
    }

    /*
     * The destructor.
     */
    ~MaskRCNN()
    {
        Py_XDECREF(mrcnn_module);
        Py_XDECREF(visualize_module);
        Py_XDECREF(load_inference_model_func);

        mrcnn_module = nullptr;
        visualize_module = nullptr;
        load_inference_model_func = nullptr;
    }

    /*
     * This is for checking our module was loaded successfully.
     */
    bool IsValid() const
    {
        return mrcnn_module != nullptr && visualize_module != nullptr && load_inference_model_func != nullptr;
    }

    /*
     * This is for loading inference model.
     */
    bool LoadInferenceModel(const char* TrainedDataPath, PyObject*& TestModel, PyObject*& InferenceConfig)
    {
        PyObject* resultInference = PyObject_CallObject(load_inference_model_func, PyTuple_Pack(2, Py_BuildValue("i", 1), Py_BuildValue("s", TrainedDataPath)));
        if (resultInference == nullptr)
        {
            std::cout << "load_inference_model() not worked as well." << std::endl;
            return false;
        }

        TestModel = PyTuple_GetItem(resultInference, 0);
        InferenceConfig = PyTuple_GetItem(resultInference, 1);
        return true;
    }

protected:
    PyObject* mrcnn_module = nullptr;
    PyObject* visualize_module = nullptr;
    PyObject* load_inference_model_func = nullptr;
};

class OpenCV
{
public:
    /*
     * The constructor.
     */
    OpenCV()
    {
        cv2_module = PyImport_ImportModule("cv2");

        if (!cv2_module) {
            std::cout << "modules are not found!" << std::endl;
            cv2_module = nullptr;
            return;
        }

        // Initialize functions.
        imread_func = PyObject_GetAttrString(cv2_module, "imread");
        if (imread_func == nullptr || !PyCallable_Check(imread_func)) {
            std::cout << "cv2.imread() not found or not callable." << std::endl;
            return;
        }
    }

    /*
     * The destructor.
     */
    ~OpenCV()
    {
        Py_XDECREF(cv2_module);
        Py_XDECREF(imread_func);

        cv2_module = nullptr;
        imread_func = nullptr;
    }

    /*
     * This is for checking our module was loaded successfully.
     */
    bool IsValid() const
    {
        return cv2_module != nullptr && imread_func != nullptr;
    }

    /*
     * This is for reading images. for more information, check opencv-python module and imread function.
     */
    bool ImRead(const char* InputImagePath, PyObject*& Image)
    {
        Image = PyObject_CallObject(imread_func, PyTuple_Pack(1, Py_BuildValue("s", InputImagePath)));
        if (Image == nullptr)
        {
            std::cout << "cv2.imread() not worked as well." << std::endl;
            return false;
        }

        return true;
    }

protected:
    PyObject* cv2_module = nullptr;
    PyObject* imread_func = nullptr;
};

int main()
{
    Py_Initialize();

    // Load libraries.
    MaskRCNN* mrcnn = new MaskRCNN;
    OpenCV* cv2 = new OpenCV;

    if (!mrcnn->IsValid() || !cv2->IsValid())
    {
        // Modules didn't loaded! We can't continue with that issue.
        return EXIT_FAILURE;
    }

    // Get image value.
    PyObject* img;
    if (!cv2->ImRead("Images\\image.jpg", img))
    {
        // There's a error.
        return EXIT_FAILURE;
    }

    // Get inference configuration.
    PyObject* test_model;
    PyObject* inference_config;
    if (!mrcnn->LoadInferenceModel("Data\\mask_rcnn_object_0005.h5", test_model, inference_config))
    {
        // There's a error.
        return EXIT_FAILURE;
    }

    Py_Finalize();

    return 0;
}
