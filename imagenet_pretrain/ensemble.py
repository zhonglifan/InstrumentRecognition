from tqdm import tqdm
import torch
import subprocess
# https://github.com/YuanGongND/psla/blob/46a53b9f86c95faae73ebd38777e2a6c370dd877/src/traintest.py


def wa(exp_dir, start_epoch, end_epoch):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sdA = torch.load(exp_dir + '/epoch_' + str(start_epoch) + '.pth')
    sdA = sdA['model']

    model_cnt = 1
    for epoch in tqdm(range(start_epoch, end_epoch+1)):
        sdB = torch.load(exp_dir + '/epoch_' + str(epoch) + '.pth')
        sdB = sdB['model']
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
        # if args.save_model == False:
        #     os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)


    # todo
    # audio_model.load_state_dict(sdA)

    wa_save_state = {
        'model': sdA
    }

    save_path = exp_dir + '/wa_model.pth'
    torch.save(wa_save_state, save_path)

    subprocess.run(['python', 'evaluator.py', '--model', 'resnet50', '--ckpt',
                    save_path,
                    '--mode', 'test', '--seg-dur', '1']
                   )



def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        ensemble_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        ensemble_predictions = np.loadtxt(exp_dir + '/predictions/ensemble_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        ensemble_predictions = ensemble_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    ensemble_predictions = ensemble_predictions / epoch
    np.savetxt(exp_dir+'/predictions/ensemble_predictions.csv', ensemble_predictions, delimiter=',')

    stats = calculate_stats(ensemble_predictions, target)
    return stats


if __name__ == '__main__':
    wa(
        exp_dir='/home/zhong_lifan/InstrumentRecognition/exp3-real/results/resnet50-lr-4/seed-2233-bs-128-epoch-35',
        start_epoch=29,
        end_epoch=34,
    )
