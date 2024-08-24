import { removePDF, UploadedFile, uploadPDF } from "@/app/actions/pdf_actions";
import { ChevronDown, ChevronUp, Trash2, Upload, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { toast } from "sonner";

interface UploadDocumentsProps {
  onFileChange: (file: File | null) => void;
  file: File | null;
}

export const UploadDocuments = ({ onFileChange, file }: UploadDocumentsProps) => {
  const [uploading, setUploading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);

    try {
      const { file_id } = await uploadPDF(file);
      const newFile = { name: file.name, file_id };
      const storedFiles = localStorage.getItem('uploadedFiles');
      const updatedFiles = storedFiles ? JSON.parse(storedFiles).concat(newFile) : [newFile];
      localStorage.setItem('uploadedFiles', JSON.stringify(updatedFiles));
      setUploadedFiles(prev => [...prev, newFile]);
      onFileChange(null);
      toast.success('File uploaded successfully!');
    } catch (error: any) {
      toast.error('Error uploading file, try again.');
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    const storedFiles = localStorage.getItem('uploadedFiles');
    if (storedFiles) {
      setUploadedFiles(JSON.parse(storedFiles));
    }
  }, []);

  const handleRemove = async (file_id: string) => {
    setIsDeleting(true);

    const promise = async () => {
      try {
        await removePDF(file_id);
        // Update local storage
        const storedFiles = localStorage.getItem('uploadedFiles');
        if (storedFiles) {
          const updatedFiles = JSON.parse(storedFiles).filter((file: UploadedFile) => file.file_id !== file_id);
          localStorage.setItem('uploadedFiles', JSON.stringify(updatedFiles));
        }
        setUploadedFiles(prev => prev.filter(file => file.file_id !== file_id));
        return { name: 'File' };
      } catch (error: any) {
        throw new Error('Error removing file, try again.');
      } finally {
        setIsDeleting(false);
      }
    };

    toast.promise(promise, {
      loading: 'Removing...',
      success: (data) => `${data.name} removed successfully!`,
      error: (error) => error.message,
    });
  };

  return (
    <div className="p-4 sm:pb-4 pb-2 bg-white border-b border-gray-200">
      <div className="max-w-3xl mx-auto">
        <Button
          className="w-full text-lg flex justify-between items-center"
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        >
          Upload PDF
          {isDropdownOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>
        
        {isDropdownOpen && (
          <div className="mt-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-2">
              <Input
                type="file"
                onChange={(e) => onFileChange(e.target.files ? e.target.files[0] : null)}
                className="hidden"
                id="file-upload"
                disabled={uploading || !!file || isDeleting}
              />
              <label
                htmlFor="file-upload"
                className={`flex flex-col items-center justify-center cursor-pointer ${file ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                <Upload className="h-8 w-8 text-gray-400 mb-2" />
                <span className="text-sm text-gray-600">Click to upload or drag and drop</span>
              </label>
            </div>
            {file && (
              <div className="mt-2 flex justify-between items-center">
                <p className="text-sm text-gray-600">Selected file: {file.name}</p>
                <Button variant="ghost" size="sm" onClick={() => onFileChange(null)} disabled={uploading || isDeleting}>
                  <X className="h-4 w-4 text-red-500" />
                </Button>
              </div>
            )}
            <Button
              className="w-full mt-2"
              onClick={handleUpload}
              disabled={uploading || !file || isDeleting}
            >
              {uploading ? "Uploading..." : <><Upload className="h-4 w-4 mr-2" />Upload</>}
            </Button>
          </div>
        )}
        
        <div className="mt-4 pt-4 border-t-2 border-neutral-500">
          <h3 className="text-lg font-semibold">Uploaded Files</h3>
          <ul className="list-disc list-inside">
            {uploadedFiles.map(({ name, file_id }, index) => (
              <li key={index} className="flex justify-between items-center text-sm text-gray-700">
                {name}
                <Button variant="ghost" size="sm" onClick={() => handleRemove(file_id)} disabled={isDeleting}>
                  <Trash2 className="h-4 w-4 text-red-500" />
                </Button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
