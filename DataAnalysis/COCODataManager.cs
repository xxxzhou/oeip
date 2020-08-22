using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using OeipCommon;

namespace DataAnalysis
{
    [Serializable]
    public class info
    {
        public int year = 0;
        public string version = string.Empty;
        public string description = string.Empty;
        public string contributor = string.Empty;
        public string url = string.Empty;
        public DateTime data_created = DateTime.MinValue;
    }

    [Serializable]
    public class image
    {
        public int id = 0;
        public int width = 0;
        public int height = 0;
        public string file_name = string.Empty;
        public int license = 0;
        public string flickr_url = string.Empty;
        public string coco_url = string.Empty;
        public DateTime data_captured = DateTime.MinValue;
    }

    [Serializable]
    public class license
    {
        public int id = 0;
        public string name = string.Empty;
        public string url = string.Empty;
    }
    //标签
    [Serializable]
    public class annotationOD
    {
        public long id = 0;
        public int image_id = 0;
        public int category_id = 0;
        //segmentation
        //public JArray segmentation = new JArray();
        //public object segmentation = new object();
        public float area = 0.0f;
        //[x,y,width,height]
        public float[] bbox = new float[4];
        public bool iscrowd = false;
    }
    //分类
    [Serializable]
    public class categorieOD
    {
        public int id = 0;
        public string name = string.Empty;
        public string supercategory = string.Empty;
    }

    [Serializable]
    public class instances
    {
        public info info = new info();
        public List<image> images = new List<image>();
        public List<license> licenses = new List<license>();
        public List<annotationOD> annotations = new List<annotationOD>();
        public List<categorieOD> categories = new List<categorieOD>();
    }

    public class Box
    {
        public float xcenter = 0.0f;
        public float ycenter = 0.0f;

        public float width = 0.0f;
        public float height = 0.0f;
    }

    public class BoxIndex
    {
        public int catId = 0;
        public Box box = new Box();
    }

    public class ImageLabel
    {
        public int imageId = 0;
        public string name = string.Empty;
        public List<BoxIndex> boxs = new List<BoxIndex>();
    }

    public class COCODataManager : MSingleton<COCODataManager>
    {
        public string logFile = string.Empty;

        protected override void Init()
        {
            logFile = Path.Combine(Application.StartupPath, "MRCoreLog.txt");
        }

        public instances LoadInstance(string path)
        {
            instances instance = new instances();
            if (File.Exists(path))
            {
                var jsonTex = File.ReadAllText(path);
                instance = JsonConvert.DeserializeObject<instances>(jsonTex);
            }
            return instance;
        }

        public async Task<instances> LoadInstanceAsync(string path)
        {
            instances instance = new instances();
            if (File.Exists(path))
            {
                var jsonTex = await Task.FromResult(File.ReadAllText(path));
                instance = await Task.FromResult(JsonConvert.DeserializeObject<instances>(jsonTex));
            }
            return instance;
        }

        public int findCategoryId(List<categorieOD> categories, int category_id)
        {
            return categories.FindIndex(p => p.id == category_id);
        }

        /// <summary>
        /// 数据经过funcFilterLabel过滤，过滤后的数据需要全部满足discardFilterLabel
        /// </summary>
        /// <param name="instData"></param>
        /// <param name="funcFilterLabel">满足条件就采用</param>
        /// <param name="discardFilterLabel">需要所有标签满足的条件</param>
        /// <returns></returns>
        public List<ImageLabel> CreateYoloLabel(instances instData, 
            Func<annotationOD, image, bool> funcFilterLabel,
            Func<annotationOD, image, bool> discardFilterLabel)
        {
            List<ImageLabel> labels = new List<ImageLabel>();
            //foreach (var image in instData.images)
            Parallel.ForEach(instData.images, (image image) =>
             {
                 var anns = instData.annotations.FindAll(p => p.image_id == image.id && funcFilterLabel(p, image));
                 bool bReserved = anns.TrueForAll((annotationOD ao) => discardFilterLabel(ao, image));
                 if (anns.Count > 0 && bReserved)
                 {
                     ImageLabel iml = new ImageLabel();
                     iml.imageId = image.id;
                     iml.name = image.file_name;
                     float dw = 1.0f / image.width;
                     float dh = 1.0f / image.height;
                     foreach (var ann in anns)
                     {
                         BoxIndex boxIndex = new BoxIndex();
                         boxIndex.box.xcenter = (ann.bbox[0] + ann.bbox[2] / 2.0f) * dw;
                         boxIndex.box.ycenter = (ann.bbox[1] + ann.bbox[3] / 2.0f) * dh;

                         boxIndex.box.width = ann.bbox[2] * dw;
                         boxIndex.box.height = ann.bbox[3] * dh;
                         //注册
                         boxIndex.catId = findCategoryId(instData.categories, ann.category_id);
                         if (boxIndex.catId >= 0)
                             iml.boxs.Add(boxIndex);
                     }
                     if (iml.boxs.Count > 0)
                     {
                         lock (labels)
                         {
                             labels.Add(iml);
                         }
                     }
                 }

             });
            return labels;
        }

        public void WriteMessage(string message, bool time = true)
        {
            if (string.IsNullOrEmpty(message))
                return;
            using (var file = new StreamWriter(logFile, true))
            {
                string msg = message;
                if (time)
                {
                    msg = "time:" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss ") + message;
                }
                file.WriteLine(msg);
            }
        }

        public void WriteMessage(Exception e, bool time = true)
        {
            if (e == null)
                return;
            using (var file = new StreamWriter(logFile, true))
            {
                string msg = "error:" + e.Message + " StackTrace:" + e.StackTrace;
                if (time)
                {
                    msg = "time:" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + msg;
                }
                file.WriteLine(msg);
            }
        }
    }
}
