def predict_test(X_test,y_test,model):
	from sklearn import metrics
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	
	y_pred_test = model.predict(X_test)
	#y_pred_test_prob = model.predict_proba(X_test)
	y_pred_test_prob = model.predict_proba(X_test)[:, 1]
	print("#####################")
	print("Test data")
	print("#####################")
	print("F1: ", metrics.f1_score(y_test, y_pred_test))
	print("Cohen Kappa: ", metrics.cohen_kappa_score(y_test, y_pred_test))
	print("Brier: ", metrics.brier_score_loss(y_test, y_pred_test))
	print("LogLoss: ", metrics.log_loss(y_test, y_pred_test_prob))
	print("ROC_AUC: ",metrics.roc_auc_score(y_test, y_pred_test_prob))

	# ROC_AUC - IMPORTANT: first argument is true values, second argument is predicted probabilities
	fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test_prob)
	plt.plot(fpr, tpr)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.title('ROC curve for stroke classifier')
	plt.xlabel('False Positive Rate (1 - Specificity)')
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.grid(True)
	plt.show()

	# PR - IMPORTANT: first argument is true values, second argument is predicted probabilities
	prec, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_test_prob)
	plt.plot(recall, prec)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.title('PR curve for stroke classifier')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.grid(True)
	plt.show()
	
	print("Classification Report :")
	print("---------------------")
	print(metrics.classification_report(y_test, y_pred_test))

	
	
	# Confusion Matrix
	print("Confusion Matrix :")
	print("----------------")
	cm = metrics.confusion_matrix(y_test,y_pred_test)
	#pd.DataFrame(conf_matrix, columns=['0 (Predicted)','1 (Predicted)'],index=['0 (Actual)','1 (Actual)'])
	df_cm = pd.DataFrame(cm, index = ['Nostroke', 'stroke'], columns = ['Nostroke', 'stroke'])	
	fig, ax = plt.subplots(figsize = (3.5, 3))
	ax = plt.axes()
	sns.heatmap(df_cm, annot=True, fmt=".0f", cmap='Blues') #, annot_kws={"size": 18})
	sns.set(font_scale=2.5)
	plt.style.use('ggplot')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	ax.set_ylabel('True Result')
	ax.set_xlabel('Predicted Result')
	plt.tight_layout()
	sns.reset_orig
	mpl.rcParams.update(mpl.rcParamsDefault)
	