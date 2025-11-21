// web-frontend/src/components/settings/UserProfile.js
import React, { useState } from "react";

const UserProfile = ({ user, onUpdate }) => {
  const [name, setName] = useState(user ? user.name : "");
  const [email, setEmail] = useState(user ? user.email : "");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onUpdate) {
      onUpdate({ name, email });
    }
  };

  if (!user) {
    return <p>Loading user profile...</p>;
  }

  return (
    <div
      className="user-profile-settings"
      style={{ padding: "20px", border: "1px solid #eee", borderRadius: "8px" }}
    >
      <h4>User Profile</h4>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "15px" }}>
          <label
            htmlFor="profile-name"
            style={{ display: "block", marginBottom: "5px" }}
          >
            Name:
          </label>
          <input
            type="text"
            id="profile-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{ width: "100%", padding: "8px", boxSizing: "border-box" }}
          />
        </div>
        <div style={{ marginBottom: "15px" }}>
          <label
            htmlFor="profile-email"
            style={{ display: "block", marginBottom: "5px" }}
          >
            Email:
          </label>
          <input
            type="email"
            id="profile-email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{ width: "100%", padding: "8px", boxSizing: "border-box" }}
          />
        </div>
        <button
          type="submit"
          style={{
            padding: "10px 15px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "4px",
          }}
        >
          Update Profile
        </button>
      </form>
    </div>
  );
};

export default UserProfile;
