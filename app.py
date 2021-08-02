import os
from functions import *
from time import time
from flask import Flask, request, render_template, send_from_directory

__author__ = 'ibininja'

app = Flask(__name__)
model = load_model('C:/Users/Юрий/PycharmProjects/pythonProject1/cash_model.h5', compile=False)
# model_coin = load_model('C:/Users/Юрий/PycharmProjects/pythonProject1/cash_model.h5', compile=False)
features1=np.load('C:/Users/Юрий/PycharmProjects/pythonProject1/color_resize.npy')
# features1=np.load('C:/Users/Юрий/PycharmProjects/pythonProject1/embeddings_coin.npy')
df_total=pd.read_csv('C:/Users/Юрий/PycharmProjects/pythonProject1/banknote.csv')
# df_total=pd.read_csv('C:/Users/Юрий/PycharmProjects/pythonProject1/catalog_coin.csv')
train_path='image1/'
# train_path='coins/'
print('df_total',df_total.shape)

folder='images/'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

print("APP_ROOT is ", APP_ROOT)
@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    time_0=time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        print(file_path)
        os.remove(file_path)

    target = os.path.join(APP_ROOT, 'images/')
    print('target is ', target)
    if not os.path.isdir(target):
            os.mkdir(target)
    # else:
        # print("Couldn't create upload directory: {}".format(target))
    # print(request.files.getlist("file"))
    uploaded=[]
    test_images=[]
    for upload in request.files.getlist("file"):
        print('upload ',upload)
        print("{} is the file name".format(upload.filename))
        print('request.files.getlist("file") ',request.files.getlist("file"))
        filename = upload.filename
        destination = "/".join([target, filename])
        uploaded.append(upload.filename)
        test_images.append(folder+upload.filename)
        # print("Accept incoming file:", filename)
        # print ("Save it to:", destination)
        upload.save(destination)
    # print('embs = ',combine_embs(folder,model).shape)
    # print('project_pca = ', project_pca(folder, model,pca,features).shape)
    # test_images=create_emb(folder,model)[1]
    print('test_images', test_images)
    time_1 = time()
    # pca_features = project_pca(folder, model,pca,features)

    features=combine_embs(model,features1,coin=0,path=folder)
    print('features = ',features.shape)
    time_2 = time()
    # idx_closest = get_closest_images(pca_features,pca_features[-1],delta=1000, num_results=10)
    # idx_closest = get_closest_images(features, features[-1], delta=1000, num_results=10)
    idx_closest = get_closest_images(features, num_results=10)
    time_3 = time()
    print('idx_closest', idx_closest)
    get_concatenated_images(idx_closest,test_images,train_path,df_total, 200)
    time_4 = time()
    print('time1 = ',time_1-time_0)
    print('time2 = ', time_2 - time_1)
    print('time3 = ', time_3 - time_2)
    print('time4 = ', time_4 - time_3)
    for i in uploaded:
        file_path = os.path.join(folder, i)
        # print(file_path)
        os.remove(file_path)
    return render_template("complete.html", image_name=filename)



@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    # print('request.files.getlist ',request.files.getlist("file"))
    print("image_names",image_names)
    return render_template("gallery.html", image_names=image_names)

@app.route("/edit", methods=["GET", "POST"])
def edit():
    global df_total
    global features1
    global train_path
    print('df_total.shape initial' , df_total.shape)
    print('features1.shape initial', features1.shape)
    if request.method == "POST":

        added = request.form.get("added")
        replaced = request.form.get("replaced")
        deleted = request.form.get("deleted")

        if len(added)>0:
          print('len added > 0')
          added_list=added.split(' ')
          for i in added_list:
            df_total=pd.concat([df_total,pd.DataFrame([i],columns=['catalog'])],axis=0,ignore_index=True)
            new_emb=combine_embs(model,features1,ind=i,train_path=train_path)
            features1=np.concatenate([features1,new_emb])
          print(df_total.shape, df_total)
          df_total.to_csv('C:/Users/Юрий/PycharmProjects/pythonProject1/banknote.csv',index=False)
          print(features1.shape, features1)
          np.save('C:/Users/Юрий/PycharmProjects/pythonProject1/color_resize.npy',features1)
    #     a=1
        # Alternatively
        if len(deleted)>0:
            print('len deleted > 0')
            deleted_list = deleted.split(' ')
            print('deleted_list ',deleted_list)

            for i in deleted_list:
                ind=df_total[df_total['catalog'] == int(i)].index
                df_total.drop(index=ind, inplace=True)
                df_total.reset_index(drop=True,inplace=True)
                ind=ind[0]
                features1 = np.concatenate([features1[:ind,:],features1[ind+1:,:]])
            print(df_total.shape, df_total)
            df_total.to_csv('C:/Users/Юрий/PycharmProjects/pythonProject1/banknote.csv', index=False)
            print(features1.shape, features1)
            np.save('C:/Users/Юрий/PycharmProjects/pythonProject1/color_resize.npy', features1)

        if len(replaced) > 0:
                print('len replaced > 0')
                replaced_list = replaced.split(' ')
                for i in replaced_list:
                    ind = df_total[df_total['catalog'] == int(i)].index
                    ind = ind[0]

                    new_emb = combine_embs(model, features1, ind=i, train_path=train_path)
                    features1[ind] = new_emb

                print(features1.shape, features1)
                np.save('C:/Users/Юрий/PycharmProjects/pythonProject1/color_resize.npy', features1)
        # username = request.form["username"]
        # email = request.form["email"]
        # password = request.form["password"]

        print(added)
        print(len(added))
        print(type(added))

        print(replaced)
        print(deleted)
        # return added,email,password

    return render_template("edit.html")

if __name__ == "__main__":
    app.run(port=1112, debug=True)
