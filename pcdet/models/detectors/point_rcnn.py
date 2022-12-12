from .detector3d_template import Detector3DTemplate
import numpy as np

class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # if True:
        #     gt_=[]
        #     #for GT box visualization in forward 
        #     # lwh, heading, xyz <= st3d
        #     # where, xyz,lwh,heading
        #     gt_box = batch_dict['gt_boxes'].squeeze(0).cpu().detach().numpy()
        #     #gt_box = [[gt_box[0][3], gt_box[0][4], gt_box[0][5]], gt_box[0][6], [gt_box[0][0], gt_box[0][1], gt_box[0][2]]] 
        #     points = batch_dict['points'].cpu().detach().numpy()
        #     np.save("/home/changwon/data/ROS/husky_prediction_bag/visualization_test/{}.npy".format(batch_dict['frame_id'].item().split(".")[0]), points)
        #     file = open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w")
        #     with open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w") as f:
        #         for num in range(gt_box.shape[0]):
        #             f.writelines("{},{},{},{},{},{},{},".format(gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]))
        #             gt_.append([gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]])
        #     #scene_viz(gt_box, points)
        #     token = batch_dict['metadata'][0]['token']
        #     print(batch_dict['frame_id'].item().split(".")[0])
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # points = batch_dict['points'].cpu().detach().numpy()
            # np.save("/home/changwon/data/ROS/husky_prediction_bag/visualization_test/{}.npy".format(batch_dict['frame_id'].item().split(".")[0]), points)
            # file = open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w")
            # with open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w") as f:
            #     for num in range(pred_dicts[0]['pred_boxes'].shape[0]):
            #         f.writelines("{},{},{},{},{},{},{},".format(pred_dicts[0]['pred_boxes'][num][3],pred_dicts[0]['pred_boxes'][num][4],pred_dicts[0]['pred_boxes'][num][5],pred_dicts[0]['pred_boxes'][num][6],pred_dicts[0]['pred_boxes'][num][0],pred_dicts[0]['pred_boxes'][num][1],pred_dicts[0]['pred_boxes'][num][2]))
            
            points = batch_dict['points'].cpu().detach().numpy()
            bbox_ = batch_dict['batch_box_preds'].squeeze().cpu().detach().numpy()
            np.save("/home/changwon/data/ROS/husky_prediction_bag/visualization_test/{}.npy".format(batch_dict['frame_id'].item().split(".")[0]), points)
            file = open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w")
            with open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w") as f:
                for num in range(bbox_.shape[0]):
                    f.writelines("{},{},{},{},{},{},{},".format(bbox_[num][3],bbox_[num][4],bbox_[num][5],bbox_[num][6],bbox_[num][0],bbox_[num][1],bbox_[num][2]))

            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
