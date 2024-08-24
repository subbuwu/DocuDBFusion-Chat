import axios from 'axios';

export interface UploadedFile {
  name: string;
  file_id: string;
}

export const uploadPDF = async (file: File): Promise<{ file_id: string }> => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await axios.post("http://127.0.0.1:8000/upload-pdf", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data; // Assuming the backend returns an object with file_id
  } catch (error) {
    console.error("Error uploading PDF:", error);
    throw new Error("Failed to upload the file.");
  }
};

export const removePDF = async (file_id: string): Promise<void> => {
  try {
    await axios.delete(`http://127.0.0.1:8000/remove-pdf/${file_id}`);
  } catch (error) {
    console.error("Error removing PDF:", error);
    throw new Error("Failed to remove the file.");
  }
};
