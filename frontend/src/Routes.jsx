import React from "react";
import { BrowserRouter, Routes as RouterRoutes, Route } from "react-router-dom";
import ScrollToTop from "components/ScrollToTop";
import ErrorBoundary from "components/ErrorBoundary";
import NotFound from "pages/NotFound";
import InterviewTranscriptionAnalysis from './pages/interview-transcription-analysis';
import JobPostingCreation from './pages/job-posting-creation';
import RecruitmentAnalyticsDashboard from './pages/recruitment-analytics-dashboard';
import VideoInterviewRoom from './pages/video-interview-room';
import AIResumeAnalysis from './pages/ai-resume-analysis';
import CandidateManagement from './pages/candidate-management';

const Routes = () => {
  return (
    <BrowserRouter>
      <ErrorBoundary>
      <ScrollToTop />
      <RouterRoutes>
        {/* Define your route here */}
        <Route path="/" element={<AIResumeAnalysis />} />
        <Route path="/interview-transcription-analysis" element={<InterviewTranscriptionAnalysis />} />
        <Route path="/job-posting-creation" element={<JobPostingCreation />} />
        <Route path="/recruitment-analytics-dashboard" element={<RecruitmentAnalyticsDashboard />} />
        <Route path="/video-interview-room" element={<VideoInterviewRoom />} />
        <Route path="/ai-resume-analysis" element={<AIResumeAnalysis />} />
        <Route path="/candidate-management" element={<CandidateManagement />} />
        <Route path="*" element={<NotFound />} />
      </RouterRoutes>
      </ErrorBoundary>
    </BrowserRouter>
  );
};

export default Routes;
