using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using Tensor = System.Numerics.Tensors.Tensor;

namespace WFInfo_onnx
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var image = LoadImage("testfiles/test3.png");
            var result = Run(image);
            Console.WriteLine(result);
        }
        public static IEnumerable<float> Run(DenseTensor<float> inputTensor)
        {

            // Get path to model to create inference session.
            var modelPath = "models/recognizer.onnx";

            // create input tensor (nlp example)
            // var inputTensor = new DenseTensor<string>(new string[] { "" }, new int[] { 1, 1 });

            // Create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input", inputTensor) };

            // Create an InferenceSession from the Model Path.
            var session = new InferenceSession(modelPath);

            // Run session and send input data in to get inference output. Call ToList then get the Last item. Then use the AsEnumerable extension method to return the Value result as an Enumerable of NamedOnnxValue.
            var outputRaw = session.Run(input);
            var output = outputRaw.ToList().Last();

            // From the Enumerable output create the inferenceResult by getting the First value and using the AsDictionary extension method of the NamedOnnxValue.
            var inferenceResult = output.AsEnumerable<float>();

            // Return the inference result as json.
            return inferenceResult;
        }

        public static DenseTensor<float> LoadImage(string path)
        {
            var image = OpenCvSharp.Cv2.ImRead(path, ImreadModes.Grayscale);

            var targetHeight = 64;
            var maxWidth = 300;

            var width = image.Width;
            var height = image.Height;
            var data = new DenseTensor<float>(new[] { 1, 1, width, height});

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    data[0, 0, x, y] = image.Get<float>(x, y);
                }
            }

            return data;

        }
 
    }
}