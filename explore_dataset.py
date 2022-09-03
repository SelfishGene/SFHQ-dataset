import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob
import clip
import torch
import pickle
import matplotlib

#%% display random subset of images from sample images


datasets_root_folder = '/home/Datasets/' # change to where dataset was downloaded

dataset_path_part_1 = os.path.join(datasets_root_folder, 'SFHQ_part1')
dataset_path_part_2 = os.path.join(datasets_root_folder, 'SFHQ_part2')

display_background_color = '0.05'
text_color = '1.0'
title_fontsize = 16

matplotlib.rcParams['text.color'] = text_color
matplotlib.rcParams['font.size'] = title_fontsize

dataset_part_to_choose = np.random.choice([1, 2])

if dataset_part_to_choose == 1:
    dataset_path = dataset_path_part_1
    sample_images_tiny_folder  = os.path.join(dataset_path, 'tiny sample (30 images)')
    sample_images_small_folder = os.path.join(dataset_path, 'small sample (550 images)')
elif dataset_part_to_choose == 2:
    dataset_path = dataset_path_part_2
    sample_images_tiny_folder  = os.path.join(dataset_path, 'a tiny sample (140 images)')
    sample_images_small_folder = os.path.join(dataset_path, 'a small sample (650 images)')

all_images_folder          = os.path.join(dataset_path, 'images')
pretrained_features_folder = os.path.join(dataset_path, 'pretrained_features')
landmarks_folder           = os.path.join(dataset_path, 'landmarks')
segmentations_folder       = os.path.join(dataset_path, 'segmentations')

num_rows = 4
num_cols = 8
num_images = num_rows * num_cols
title_fontsize = 16

if num_images <= 30:
    sample_image_filenames = glob.glob(os.path.join(sample_images_tiny_folder, '*.jpg'))
else:
    sample_image_filenames = glob.glob(os.path.join(sample_images_small_folder, '*.jpg'))

selected_images = np.random.choice(sample_image_filenames, size=num_images, replace=False)

plt.close('all')
fig = plt.figure(figsize=(40,30))
fig.patch.set_facecolor(display_background_color)
fig.subplots_adjust(left=0.003, right=0.997, bottom=0.003, top=0.99, hspace=0.02, wspace=0.02)
for k, curr_image_filename in enumerate(selected_images):
    curr_image = PIL.Image.open(curr_image_filename).convert("RGB")

    plt.subplot(num_rows, num_cols, k + 1); plt.imshow(curr_image); plt.axis('off')
    plt.title('image "%s"' %(curr_image_filename.split('/')[-1].split('.')[0]), fontsize=title_fontsize)


#%% display random subset of images from the data along with their landmarks

num_images_to_show = 6

num_rows = 3
num_cols = num_images_to_show

sample_image_filenames = glob.glob(os.path.join(all_images_folder, '*.jpg'))
selected_images = np.random.choice(sample_image_filenames, size=num_images_to_show, replace=False)

title_fontsize = 16

plt.close('all')
fig = plt.figure(figsize=(40,30))
fig.patch.set_facecolor(display_background_color)
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.98, hspace=0.03, wspace=0.04)
for k, curr_image_filename in enumerate(selected_images):

    curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]

    curr_image_filename = os.path.join(all_images_folder, curr_sample_name + '.jpg')
    curr_landmarks_filename = os.path.join(landmarks_folder, curr_sample_name + '.npz')
    curr_segmentation_filename = os.path.join(segmentations_folder, curr_sample_name + '.png')

    curr_image = PIL.Image.open(curr_image_filename).convert("RGB")
    curr_landmarks = np.load(curr_landmarks_filename)['landmarks']
    # curr_segmentation = imageio.imread(curr_segmentation_filename)
    curr_segmentation = PIL.Image.open(curr_segmentation_filename).convert("L")

    plt.subplot(num_rows, num_cols, k + 1 + 0 * num_cols)
    plt.imshow(curr_image); plt.axis('off')
    plt.title('image "%s"' %(curr_sample_name), fontsize=title_fontsize)

    plt.subplot(num_rows, num_cols, k + 1 + 1 * num_cols)
    plt.imshow(curr_image); plt.axis('off')
    plt.scatter(curr_landmarks[:,0],curr_landmarks[:,1], c='r')
    plt.title('image with landmarks overlayed', fontsize=title_fontsize)

    plt.subplot(num_rows, num_cols, k + 1 + 2 * num_cols)
    plt.imshow(curr_segmentation); plt.axis('off')
    plt.scatter(curr_landmarks[:,0],curr_landmarks[:,1], c='r')
    plt.title('segmentation mask with landmarks overlayed', fontsize=title_fontsize)


#%% demonstate segmentation overlay

num_images_to_show = 4
num_cols = 8
num_rows = num_images_to_show

sample_image_filenames = glob.glob(os.path.join(all_images_folder, '*.jpg'))
sample_image_filenames = glob.glob(os.path.join(sample_images_small_folder, '*.jpg'))
selected_images = np.random.choice(sample_image_filenames, size=num_images_to_show, replace=False)

darkening_mult_factor = 0.35
title_fontsize = 18

plt.close('all')
fig = plt.figure(figsize=(40,30))
fig.patch.set_facecolor(display_background_color)
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, hspace=0.04, wspace=0.03)
for k, curr_image_filename in enumerate(selected_images):
    curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]

    curr_image_filename = os.path.join(all_images_folder, curr_sample_name + '.jpg')
    curr_segmentation_filename = os.path.join(segmentations_folder, curr_sample_name + '.png')

    curr_image = np.array(PIL.Image.open(curr_image_filename).convert("RGB"))
    curr_segmentation = np.array(PIL.Image.open(curr_segmentation_filename).convert("L"))

    only_face = (curr_segmentation >= 1) & (curr_segmentation <= 13)
    edited_face_1 = curr_image.copy()
    edited_face_1[~only_face] = darkening_mult_factor * edited_face_1[~only_face]

    only_face_skin = (curr_segmentation == 1)
    edited_face_2 = curr_image.copy()
    edited_face_2[~only_face_skin] = darkening_mult_factor * edited_face_2[~only_face_skin]

    only_face_parts = (curr_segmentation > 1) & (curr_segmentation <= 13)
    edited_face_3 = curr_image.copy()
    edited_face_3[~only_face_parts] = darkening_mult_factor * edited_face_3[~only_face_parts]

    only_background_neck_and_shirt = (curr_segmentation == 0) | ((curr_segmentation >= 14) & (curr_segmentation <= 16))
    edited_face_4 = curr_image.copy()
    edited_face_4[only_background_neck_and_shirt] = darkening_mult_factor * edited_face_4[only_background_neck_and_shirt]

    only_hair_and_hats = (curr_segmentation >= 17)
    edited_face_5 = curr_image.copy()
    edited_face_5[~only_hair_and_hats] = darkening_mult_factor * edited_face_5[~only_hair_and_hats]

    only_background_neck_and_shirt = (curr_segmentation == 0) | ((curr_segmentation >= 14) & (curr_segmentation <= 16))
    edited_face_6 = curr_image.copy()
    edited_face_6[~only_background_neck_and_shirt] = darkening_mult_factor * edited_face_6[~only_background_neck_and_shirt]

    plt.subplot(num_rows, num_cols, k * num_cols + 1); plt.imshow(curr_image); plt.axis('off'); plt.title('original "%s"' %(curr_sample_name), fontsize=title_fontsize)
    plt.subplot(num_rows, num_cols, k * num_cols + 2); plt.imshow(curr_segmentation); plt.axis('off'); plt.title('face parsing', fontsize=title_fontsize)
    plt.subplot(num_rows, num_cols, k * num_cols + 3); plt.imshow(edited_face_1); plt.axis('off'); plt.title('face only', fontsize=title_fontsize)
    plt.subplot(num_rows, num_cols, k * num_cols + 4); plt.imshow(edited_face_2); plt.axis('off'); plt.title('skin only', fontsize=title_fontsize)
    plt.subplot(num_rows, num_cols, k * num_cols + 5); plt.imshow(edited_face_3); plt.axis('off'); plt.title('face parts', fontsize=title_fontsize)
    plt.subplot(num_rows, num_cols, k * num_cols + 6); plt.imshow(edited_face_4); plt.axis('off'); plt.title('face and hair', fontsize=title_fontsize)
    plt.subplot(num_rows, num_cols, k * num_cols + 7); plt.imshow(edited_face_5); plt.axis('off'); plt.title('hair and hats only', fontsize=title_fontsize)
    plt.subplot(num_rows, num_cols, k * num_cols + 8); plt.imshow(edited_face_6); plt.axis('off'); plt.title('background, neck and shirt', fontsize=title_fontsize)


#%% gather all CLIP ViT/14 @ 336 embeddings into a single matrix


def collect_pretrained_features_from_folder(base_image_folder, requested_features_model='CLIP_ViTL_14@336', nromalize_features=True):
    # this function assumes that the folder stucture is proper and features dict contains the requested features

    images_folder = os.path.join(base_image_folder, 'images')
    features_folder = os.path.join(base_image_folder, 'pretrained_features')
    all_feature_dict_filenames = glob.glob(os.path.join(features_folder, '*.pickle'))
    all_image_filenames = glob.glob(os.path.join(images_folder, '*.*'))

    try:
        curr_features_dict = pickle.load(open(all_feature_dict_filenames[0], "rb"))
        num_features = curr_features_dict[requested_features_model].shape[1]
    except:
        print('the requested features were not calculated.')
        return [],[]

    num_images = len(all_feature_dict_filenames)

    # create matrix to fill
    pretrained_image_features_matrix = np.zeros((num_images, num_features))

    # go over all samples and collect the features
    image_filename_map = {}
    for k, curr_image_filename in enumerate(all_image_filenames):
        curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]
        curr_features_dict_filename = os.path.join(features_folder, curr_sample_name + '.pickle')
        curr_features_dict = pickle.load(open(curr_features_dict_filename, "rb"))
        pretrained_image_features_matrix[k,:] = curr_features_dict[requested_features_model]
        image_filename_map[k] = curr_image_filename

    # normalize features to unit norm
    if nromalize_features:
        pretrained_image_features_matrix /= np.linalg.norm(pretrained_image_features_matrix, axis=1, keepdims=True)

    return pretrained_image_features_matrix, image_filename_map


# extract features for all filenames from two parts of the dataset
CLIP_image_features_pt1, image_filename_map_pt1 = collect_pretrained_features_from_folder(dataset_path_part_1, requested_features_model='CLIP_ViTL_14@336', nromalize_features=True)
CLIP_image_features_pt2, image_filename_map_pt2 = collect_pretrained_features_from_folder(dataset_path_part_2, requested_features_model='CLIP_ViTL_14@336', nromalize_features=True)

# merge the image features and filenames from both datasets for simple querying
CLIP_image_features = np.concatenate((CLIP_image_features_pt1, CLIP_image_features_pt2), axis=0)
image_filename_map = image_filename_map_pt1.copy()
for k in range(CLIP_image_features_pt2.shape[0]):
    image_filename_map[CLIP_image_features_pt1.shape[0] + k] = image_filename_map_pt2[k]

# load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, CLIP_preprocess = clip.load("ViT-L/14@336px", device=device)


#%% make some textual searches


# please uncomment desired queries (or just make up some of your own)


# hair related (color x style)
text_prefix = ''
text_strings = ['white or gray hair', 'yellow or blond hair', 'green hair', 'blue hair', 'purple or pink hair', 'red or orange hair']

# text_prefix = 'woman with '
# text_strings = ['short blond hair', 'long blond hair', 'short red hair', 'long red hair', 'short black hair', 'long black hair']

# text_prefix = ''
# text_strings = ['straight hair', 'curly hair', 'high top hairstyle', 'bob-cut hairstyle', 'afro hairstyle']


# various random properties
# text_prefix = 'woman '
# text_strings = ['heavy makeup', 'without makeup', 'red lipstick', 'strong eyeliner']

# text_prefix = ''
# text_strings = ['yellow background', 'green background', 'blue background', 'purple background', 'red background']

# text_prefix = ''
# text_strings = ['reading glasses', 'sunglasses', 'bald', 'goatee', 'lipstick']

# text_prefix = ''
# text_strings = ['large or chiseled jaw', 'long white beard', 'fashionable beard', 'long forehead', 'overweight or chubby']


# expression
# text_prefix = ''
# text_strings = ['angry or enraged', 'surprised', 'smiling', 'sad or depressed', 'grim face']

# text_prefix = 'man '
# text_strings = ['angry or enraged', 'surprised', 'smiling', 'sad or depressed', 'grim face']


# ethnicity (with age cross)
# text_prefix = ''
# text_strings = ['asian', 'indian', 'african', 'persian', 'south-american', 'irish']

# text_prefix = 'old age '
# text_strings = ['asian', 'indian', 'african', 'persian', 'south-american', 'irish']

# text_prefix = 'typical adult '
# text_strings = ['asian', 'indian', 'african', 'persian', 'south-american', 'irish']

# text_prefix = 'young child '
# text_strings = ['asian', 'indian', 'african', 'persian', 'south-american', 'irish']


# age (with ethnicity cross)
# text_prefix = ''
# text_strings = ['10 month old baby', '2.5 year old toddler', 'small child', '16 year old teenager', '30 year old adult', 'wrinkly 70 year old senior']

# text_prefix = 'asian female '
# text_strings = ['10 month old baby', '2.5 year old toddler', 'small child', '16 year old teenager', '30 year old adult', 'wrinkly 70 year old senior']

# text_prefix = 'african male '
# text_strings = ['10 month old baby', '2.5 year old toddler', 'small child', '16 year old teenager', '30 year old adult', 'wrinkly 70 year old senior']


# will randomly display "num_top_images_to_show" among the top "num_top_image_candidates" best matching queries
num_top_images_to_show = 2 * len(text_strings) + 1
num_top_image_candidates = int(3 * num_top_images_to_show)

title_fontsize = 14

# attach prefix and extract text features
text_strings_full = [(text_prefix + x) for x in text_strings]
tokenized_text_samples = torch.cat([clip.tokenize(text_strings_full)]).cuda()
CLIP_text_features = CLIP_model.encode_text(tokenized_text_samples).detach().cpu().numpy()
CLIP_text_features /= np.linalg.norm(CLIP_text_features, axis=1, keepdims=True) # normalize to unit norm

# perform inner product to get image-text similarity score
image_text_similarity  = np.dot(CLIP_image_features , CLIP_text_features.T)

num_rows = len(text_strings)
num_cols = num_top_images_to_show

plt.close('all')
fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(40, 32))
fig.patch.set_facecolor(display_background_color)
fig.subplots_adjust(left=0.003,right=0.997,bottom=0.01,top=0.925,hspace=0.13,wspace=0.03)
fig.suptitle('image textual search using CLIP features from synthetic dataset \nprefix_text = "%s"' %(text_prefix), fontsize=25)
for row_ind, q_str in enumerate(text_strings):

    # get top "num_top_image_candidates" matching queries sorted from best matching downward
    query_best_inds = list(np.argsort(image_text_similarity[:,row_ind])[-num_top_image_candidates:])
    query_best_inds.reverse()
    # randomly select "num_top_images_to_show" from that list
    query_best_inds = np.random.choice(query_best_inds, size=num_top_images_to_show, replace=False)

    for col_ind in range(num_cols):
        curr_image = PIL.Image.open(image_filename_map[query_best_inds[col_ind]]).convert("RGB")
        ax[row_ind,col_ind].imshow(curr_image); ax[row_ind,col_ind].set_axis_off()
        ax[row_ind,col_ind].set_title("'%s'" %(q_str), fontsize=title_fontsize)


#%%


