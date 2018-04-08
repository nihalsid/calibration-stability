import xmltodict
import numpy as np

def read_calibration(calibration_file):
	calibrations = []
	with open(calibration_file, "r") as f:
		parsed_xml = xmltodict.parse(f.read())
	cameras = parsed_xml["calibration"]["camera"]
	idx = 0
	for camera in cameras:
		projection = camera["projection"][0]
		K = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float)
		K[0][0] = projection['alpha']
		K[0][1] = projection['skew']
		K[0][2] = projection['principal']['x']
		K[1][0] = 0
		K[1][1] = projection['beta']
		K[1][2] = projection['principal']['y']
		K[2][0] = 0
		K[2][1] = 0
		K[2][2] = 1
		pose = camera["pose"]
		R = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float)
		R[0][0] = pose["rotation"]["matrix"]["a00"]
		R[0][1] = pose["rotation"]["matrix"]["a01"]
		R[0][2] = pose["rotation"]["matrix"]["a02"]
		R[1][0] = pose["rotation"]["matrix"]["a10"]
		R[1][1] = pose["rotation"]["matrix"]["a11"]
		R[1][2] = pose["rotation"]["matrix"]["a12"]
		R[2][0] = pose["rotation"]["matrix"]["a20"]
		R[2][1] = pose["rotation"]["matrix"]["a21"]
		R[2][2] = pose["rotation"]["matrix"]["a22"]
		T = np.array([0, 0, 0], dtype=np.float)
		T[0] = pose["translation"]["x"]
		T[1] = pose["translation"]["y"]
		T[2] = pose["translation"]["z"]
		T = T.reshape((3, 1))
		p_T = np.matmul(-1 * R, T)
		P = {
			"R": R,
			"T": p_T,
			"K": K,
			"WR": np.transpose(R),
			"WT": T
		}
		calibrations.append(P)
		print """	Matrix3d K%d, R%d, WR%d;
					Vector3d T%d, WT%d;
					K%d << %f, %f, %f, %f, %f, %f, %f, %f, %f;
					R%d << %f, %f, %f, %f, %f, %f, %f, %f, %f;
					WT%d << %f, %f, %f;
					calibrations.push_back(Calibration(K%d, R%d, WT%d));""" %(idx,idx,idx,idx,idx,idx,K[0][0],K[0][1],K[0][2],K[1][0],K[1][1],K[1][2],K[2][0],K[2][1],K[2][2],idx,R[0][0],R[0][1],R[0][2],R[1][0],R[1][1],R[1][2],R[2][0],R[2][1],R[2][2],idx,T[0],T[1],T[2],idx,idx,idx)
		idx+=1
	return calibrations
read_calibration(r"C:\Users\Yawar\Documents\PycharmProjects\PoseEstimation\Input1\Calibration.xml")