from .detector3d_template import Detector3DTemplate
import numpy as np

class CenterPoint_PointPillar_RCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        # if True:
        #     for batch in range(batch_dict['gt_boxes'].squeeze(0).cpu().detach().numpy().shape[0]):
        #         gt_=[]
        #         #for GT box visualization in forward 
        #         # where, xyz,lwh,heading
        #         gt_box = batch_dict['gt_boxes'].squeeze(0).cpu().detach().numpy()[batch]
        #         #gt_box = [[gt_box[0][3], gt_box[0][4], gt_box[0][5]], gt_box[0][6], [gt_box[0][0], gt_box[0][1], gt_box[0][2]]] 
        #         points = batch_dict['points'].cpu().detach().numpy()
        #         pc_mask = (points[:, 0] == float(batch))
        #         points = points[pc_mask]
        #         np.save("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/pc/{}.npy".format(batch_dict['frame_id'][batch].item().split(".")[0]), points)
        #         file = open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/{}.txt".format(batch_dict['frame_id'][batch].item().split(".")[0]), "w")
        #         with open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/{}.txt".format(batch_dict['frame_id'][batch].item().split(".")[0]), "w") as f:
        #             for num in range(gt_box.shape[0]):
        #                 f.writelines("{},{},{},{},{},{},{},".format(gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]))
        #                 gt_.append([gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]])
                
        #         pred_box = batch_dict['batch_box_preds'].squeeze(0).cpu().detach().numpy()[batch]
        #         with open("/home/changwon/detection_task/SSOD/kakao/my_ssda_2/vis_in_model/box/pred_{}.txt".format(batch_dict['frame_id'][batch].item().split(".")[0]), "w") as f:
        #             for num in range(pred_box.shape[0]):
        #                 f.writelines("{},{},{},{},{},{},{},".format(pred_box[num][3],pred_box[num][4],pred_box[num][5],pred_box[num][6],pred_box[num][0],pred_box[num][1],pred_box[num][2]))
        #                 gt_.append([pred_box[num][3],pred_box[num][4],pred_box[num][5],pred_box[num][6],pred_box[num][0],pred_box[num][1],pred_box[num][2]])
        #         #scene_viz(gt_box, points)
        #         #token = batch_dict['metadata'][0]['token']
        #         print(batch_dict['frame_id'][batch].item().split(".")[0])

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        #loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        # loss = loss_rpn + loss_point + loss_rcnn
        loss = loss_rpn + loss_rcnn
        #loss = loss_rpn
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        #final_pred_dict = batch_dict['final_box_dicts']
        final_pred_dict = [{'pred_boxes':batch_dict["rois"][i], 'pred_scores':batch_dict['roi_scores'][i], 'pred_labels':batch_dict['roi_labels'][i]} for i in range(batch_size)]
        #final_pred_dict = batch_dict['rois']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            #pred_boxes = final_pred_dict[index]

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict